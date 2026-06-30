import sys

print(sys.executable)
import os
import sys
import time
import datetime
import requests
import dataset
import base64
import io
import glob
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# ============================================================
# Constants and Configuration
# ============================================================
API_URL = 'https://api.lost2.petcolove.org/maps/pets-with-auth/search'
DATABASE_DIRECTORY = os.path.expanduser('~/data/databases')
DATABASE_FILE_PATH = os.path.join(DATABASE_DIRECTORY, 'found_cats.db')
STATE_FILE_PATH = os.path.join(DATABASE_DIRECTORY, 'last_run_timestamp.txt')
DATABASE_CONNECTION_URL = 'sqlite:///' + DATABASE_FILE_PATH
TABLE_NAME = 'found_cats'
LOCAL_IMAGE_DIRECTORY = '/Users/stevendao/data/found_cat/images/'

SLEEP_SECONDS_PER_REQUEST = 2
PAGE_LIMIT_COUNT = 24
PAGINATION_START_OFFSET = 0
COOLDOWN_SECONDS = 28800
MAX_CONSECUTIVE_ERRORS = 3

TARGET_GENDER_SKIP = 'female'
TARGET_STATUS_FOUND = 'found'

TARGET_STATUS_SIGHTING = 'sighting'
TARGET_DATE_YEAR = 2026
TARGET_DATE_MONTH = 4
TARGET_DATE_DAY = 1
DATE_FORMAT_STRING = "%Y-%m-%d"
DATE_PREFIX_LENGTH = 10

FILE_READ_MODE = 'r'
FILE_WRITE_MODE = 'w'
IMAGE_ENCODING_FORMAT = 'utf-8'

# ============================================================
# SCORING WEIGHTS
# Tune these to favor color vs. CLIP visual similarity.
# Both sum to 1.0 and produce a final 0–100 score.
# ============================================================
CLIP_WEIGHT = 0.55          # CLIP captures shape, texture, pose
COLOR_HIST_WEIGHT = 0.45    # Color histogram catches fur color directly

# CLIP: raw cosine similarity typically 0.70–0.99 for same-species images.
# We map the usable range [CLIP_FLOOR, 1.0] → [0, 1].
# Raising the floor makes the score more discriminating within same-species cats.
CLIP_SIMILARITY_FLOOR = 0.75   # below this = 0 (clearly different animal)
CLIP_SIMILARITY_CEIL  = 0.97   # above this = 1.0 (essentially identical)

# ============================================================
# AGGREGATION TUNING
#
# Problem: a single high-quality reference (ff0.png, score=55)
# gets diluted by many weak/noisy references (score=25-38),
# collapsing the final score to ~32 — same as a black cat.
#
# Fix: softmax-weighted mean so high-scoring references dominate,
# blended with the raw max so one strong match always lifts the score.
#
# SOFTMAX_TEMPERATURE:
#   Low  (0.05) → winner-take-all, only best reference matters
#   High (0.30) → more democratic, similar to plain average
#   Recommended: 0.08–0.12 for 5–15 mixed-quality references
#
# BEST_OF_BLEND:
#   0.0 → pure softmax-weighted mean (ignores single strong hits)
#   1.0 → pure max (ignores all other references)
#   0.35 → good balance; a 55-point hit yields ~47 final vs ~32 before
# ============================================================
SOFTMAX_TEMPERATURE = 0.10   # controls how sharply high scores dominate
BEST_OF_BLEND       = 0.35   # weight given to the single best reference score

TARGET_DATE_CUTOFF = datetime.datetime(
    year=TARGET_DATE_YEAR,
    month=TARGET_DATE_MONTH,
    day=TARGET_DATE_DAY
)

API_TOKEN_VALUE = '482f7a44f98ceec8b96ef4d72565fb2b'
API_SPECIES_VALUE = 'cat'

API_START_DATE_VALUE = '2026-04-01'

if time.time() > 1782749621 + (86400 * 10):
    API_START_DATE_VALUE = '2026-05-01'

if time.time() > 1782749621 + (86400 * 15):
    API_START_DATE_VALUE = '2026-06-01'

API_CREATED_AFTER_VALUE = '2026-04-06'
API_RADIUS_VALUE = '100'

API_LATITUDE_VALUE ='29.62431045357618'
API_LONGITUDE_VALUE = '-95.69754879999999'

LONG_LAT_LIST = [
    ('29.94463706277179', '-95.50996676695979'),
]

home_lat = 29.94463706277179
home_lon = -95.50996676695979

LONG_LAT_LIST += sorted(
    [
        (
            f"{lat + 0.62431045357618:.14f}",
            f"{lon + 0.46843140581424:.14f}",
        )
        for lat in range(25, 35)
        for lon in range(-105, -90)
    ],
    key=lambda p: (
        (float(p[0]) - home_lat) ** 2 +
        (float(p[1]) - home_lon) ** 2
    )
)

for API_LATITUDE_VALUE, API_LONGITUDE_VALUE in LONG_LAT_LIST:
    IMAGE_BASE_URL = 'https://d1xo1ei89o6wi.cloudfront.net/'
    IMAGE_DOWNLOAD_TIMEOUT_SECONDS = 10
    API_DOWNLOAD_TIMEOUT_SECONDS = 15
    MODEL_ID = "openai/clip-vit-base-patch32"
    IMAGE_MODE = "RGB"

    VALID_IMAGE_EXTENSIONS = [
        '.png',
        '.jpg',
        '.jpeg',
        '.webp',
    ]

    HEADERS = {
        'accept': 'application/json, text/plain, */*',
        'accept-language': 'en',
        'dnt': '1',
        'origin': 'https://petcolove.org',
        'priority': 'u=1, i',
        'referer': 'https://petcolove.org/',
        'sec-ch-ua': '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-site',
        'sec-gpc': '1',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
    }

    BASE_REQUEST_PARAMS = {
        'token': API_TOKEN_VALUE,
        'species': API_SPECIES_VALUE,
        'start': API_START_DATE_VALUE,
        'types': [
            'foundUserPet',
            'foundOrgPet',
            'sighting',
        ],
        'created_after': API_CREATED_AFTER_VALUE,
        'radius': API_RADIUS_VALUE,
        'latitude': API_LATITUDE_VALUE,
        'longitude': API_LONGITUDE_VALUE,
        'limit': str(PAGE_LIMIT_COUNT),
    }


    # ============================================================
    # Image Feature Helpers
    # ============================================================

    def extract_clip_embedding(pil_image: Image.Image, processor, model) -> torch.Tensor:
        """
        Extract a CLIP embedding WITHOUT cropping.

        Cropping was the original mistake: it removed fur/color from edges, which
        is exactly where the most discriminating signal lives for breed/color matching.
        CLIP's own attention handles framing internally.
        """
        inputs = processor(images=pil_image, return_tensors="pt")
        with torch.no_grad():
            raw = model.get_image_features(**inputs)

        # Handle non-Tensor outputs (older transformers versions)
        if not isinstance(raw, torch.Tensor):
            if hasattr(raw, 'image_embeds'):
                raw = raw.image_embeds
            elif hasattr(raw, 'pooler_output'):
                raw = raw.pooler_output
            else:
                raw = raw[0]

        return F.normalize(raw, p=2, dim=1)


    def build_color_histogram(pil_image: Image.Image, bins: int = 32) -> np.ndarray:
        """
        Build a normalised HSV color histogram for the image.

        Why HSV instead of RGB?
        - Hue separates color identity (orange vs. grey) cleanly.
        - Saturation separates vivid fur from washed-out backgrounds.
        - We weight Hue most heavily because it is the primary discriminator.

        Returns a 1-D float32 array (normalised to sum=1).
        """
        # Convert PIL → numpy BGR → HSV
        np_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        np_hsv = cv2.cvtColor(np_bgr, cv2.COLOR_BGR2HSV)

        # Compute per-channel histograms
        h_hist = cv2.calcHist([np_hsv], [0], None, [bins], [0, 180]).flatten()
        s_hist = cv2.calcHist([np_hsv], [1], None, [bins], [0, 256]).flatten()
        v_hist = cv2.calcHist([np_hsv], [2], None, [bins], [0, 256]).flatten()

        # Weight: Hue 60%, Saturation 30%, Value 10%
        weighted = np.concatenate([
            h_hist * 0.60,
            s_hist * 0.30,
            v_hist * 0.10,
        ])

        total = weighted.sum()
        if total > 0:
            weighted /= total

        return weighted.astype(np.float32)


    def color_histogram_similarity(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
        """
        Bhattacharyya coefficient — returns 0.0 (no overlap) to 1.0 (identical).

        Chosen over correlation or chi-squared because it is naturally bounded [0,1]
        and robust to small bin-count differences.
        """
        # Bhattacharyya coefficient = sum(sqrt(a_i * b_i))
        return float(np.sum(np.sqrt(hist_a * hist_b)))


    def clip_cosine_to_score(raw_cosine: float) -> float:
        """
        Map raw CLIP cosine similarity → [0.0, 1.0] linearly within the
        meaningful operating range [CLIP_SIMILARITY_FLOOR, CLIP_SIMILARITY_CEIL].

        Why:
        - Two unrelated cats of the same species routinely score 0.75–0.80.
        - Identical photos of the same cat score ~0.97–0.99.
        - The original code's (raw - 0.70) * 500 made differences of 0.002 in
          raw cosine matter enormously, producing random-looking 0–100 scores.
        """
        clipped = max(CLIP_SIMILARITY_FLOOR, min(CLIP_SIMILARITY_CEIL, raw_cosine))
        span = CLIP_SIMILARITY_CEIL - CLIP_SIMILARITY_FLOOR
        if span == 0:
            return 1.0
        return (clipped - CLIP_SIMILARITY_FLOOR) / span


    def compute_combined_score(
        clip_emb_query: torch.Tensor,
        color_hist_query: np.ndarray,
        clip_emb_ref: torch.Tensor,
        color_hist_ref: np.ndarray,
    ) -> float:
        """
        Combine CLIP visual similarity and color histogram similarity into a
        single 0–100 score.

        Higher = more likely to be the same cat.
        """
        # 1. CLIP cosine similarity (shape: scalar)
        raw_cosine = torch.matmul(clip_emb_query, clip_emb_ref.T).item()
        clip_score_01 = clip_cosine_to_score(raw_cosine)

        # 2. Color histogram Bhattacharyya coefficient (already 0–1)
        color_score_01 = color_histogram_similarity(color_hist_query, color_hist_ref)

        # 3. Weighted combination → scale to 0–100
        combined_01 = (CLIP_WEIGHT * clip_score_01) + (COLOR_HIST_WEIGHT * color_score_01)
        final_score = round(combined_01 * 100.0, 2)

        return final_score


    def aggregate_scores(per_ref_scores: list[float]) -> float:
        """
        Combine per-reference scores into a single 0–100 final score.

        Strategy: softmax-weighted mean blended with the best single score.

        Why not plain mean?
            With 9 references where 1 scores 55 and 8 score ~28, plain mean = 32.
            That's the same as a completely wrong cat. The good reference gets
            outvoted by noise.

        Why not plain max?
            Susceptible to a single lucky reference image that happens to look
            compositionally similar to an unrelated cat.

        Softmax-weighted mean:
            Each score s_i gets weight = exp(s_i / T) / sum(exp(s_j / T)).
            With T=0.10, a score of 55 gets ~8x the weight of a score of 28.
            The weighted mean is then pulled strongly toward 55.

        Final blend:
            final = (1 - BEST_OF_BLEND) * softmax_mean + BEST_OF_BLEND * max_score
            With BEST_OF_BLEND=0.35:
                = 0.65 * ~50 + 0.35 * 55  ≈  51.8   (vs. 32.4 before)

        This means:
            - A single strong hit (ff0.png) meaningfully lifts the final score.
            - Multiple mediocre matches don't collapse a real positive result.
            - A query cat with ALL references scoring ~28 still scores ~28 (correct).
        """
        if not per_ref_scores:
            return 0.0

        if len(per_ref_scores) == 1:
            return round(per_ref_scores[0], 2)

        scores_np = np.array(per_ref_scores, dtype=np.float64)

        # Softmax weights — subtract max for numerical stability (standard trick)
        shifted = (scores_np - scores_np.max()) / SOFTMAX_TEMPERATURE
        weights = np.exp(shifted)
        weights /= weights.sum()

        softmax_mean = float(np.dot(weights, scores_np))
        best_score   = float(scores_np.max())

        blended = (1.0 - BEST_OF_BLEND) * softmax_mean + BEST_OF_BLEND * best_score
        return round(float(np.clip(blended, 0.0, 100.0)), 2)


    # ============================================================
    # State helpers (unchanged from original)
    # ============================================================

    def check_execution_cooldown():
        current_timestamp = time.time()
        # TODO: REMOVE THIS CHECK 

        # if os.path.exists(STATE_FILE_PATH):
        #     try:
        #         with open(STATE_FILE_PATH, FILE_READ_MODE) as fh:
        #             last_run_timestamp = float(fh.read().strip())
        #         if current_timestamp - last_run_timestamp < COOLDOWN_SECONDS:
        #             print(f"Aborting. Script ran within last {COOLDOWN_SECONDS}s. last_run_timestamp={last_run_timestamp}.")
        #             sys.exit(0)
        #     except Exception as exc:
        #         print(f"Failed to read state file. Exc={exc}")
        #         raise


    def record_execution_timestamp():
        try:
            with open(STATE_FILE_PATH, FILE_WRITE_MODE) as fh:
                fh.write(str(time.time()))
            print("Recorded execution timestamp.")
        except Exception as exc:
            print(f"Failed to write state file. Exc={exc}")
            raise


    # ============================================================
    # Main
    # ============================================================

    def main():
        check_execution_cooldown()

        os.makedirs(DATABASE_DIRECTORY, exist_ok=True)
        database_client = dataset.connect(DATABASE_CONNECTION_URL)
        records_table = database_client[TABLE_NAME]

        print("Initializing CLIP model.")
        clip_processor = CLIPProcessor.from_pretrained(MODEL_ID)
        clip_model = CLIPModel.from_pretrained(MODEL_ID)

        # ----------------------------------------------------------
        # Load reference images (your missing cat photos)
        # ----------------------------------------------------------
        # Each entry: {"clip": Tensor, "color": np.ndarray, "path": str}
        reference_image_data = []

        local_image_paths = glob.glob(os.path.join(LOCAL_IMAGE_DIRECTORY, '*'))
        print(f"Scanning {len(local_image_paths)} local reference files.")

        local_consecutive_errors = 0
        for local_path in local_image_paths:
            iteration_success = False
            ext = os.path.splitext(local_path)[1].lower()
            if ext not in VALID_IMAGE_EXTENSIONS:
                print(f"Skipping non-image file. path={local_path}.")
                continue

            try:
                pil_img = Image.open(local_path).convert(IMAGE_MODE)

                clip_emb = extract_clip_embedding(pil_img, clip_processor, clip_model)
                color_hist = build_color_histogram(pil_img)

                reference_image_data.append({
                    "clip": clip_emb,
                    "color": color_hist,
                    "path": local_path,
                })

                print(f"Loaded reference image. path={local_path}.")
                iteration_success = True

            except Exception as exc:
                local_consecutive_errors += 1
                print(f"Failed to process local image. path={local_path}, errors={local_consecutive_errors}. Exc={exc}")
                if local_consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    raise
            finally:
                if iteration_success:
                    local_consecutive_errors = 0

        print(f"Loaded {len(reference_image_data)} reference images.")

        # ----------------------------------------------------------
        # Pagination loop
        # ----------------------------------------------------------
        current_pagination_offset = PAGINATION_START_OFFSET

        while True:
            request_parameters = BASE_REQUEST_PARAMS.copy()
            request_parameters['offset'] = str(current_pagination_offset)

            print(f"Requesting offset={current_pagination_offset}.")

            api_request_success = False
            api_request_errors = 0
            while api_request_errors < MAX_CONSECUTIVE_ERRORS:
                try:
                    print(f"Sleeping {SLEEP_SECONDS_PER_REQUEST}s before API request.")
                    time.sleep(SLEEP_SECONDS_PER_REQUEST)
                    api_response = requests.get(
                        url=API_URL,
                        params=request_parameters,
                        headers=HEADERS,
                        timeout=API_DOWNLOAD_TIMEOUT_SECONDS,
                    )
                    api_response.raise_for_status()
                    api_request_success = True
                    break
                except requests.exceptions.RequestException as exc:
                    api_request_errors += 1
                    print(f"API request failed. errors={api_request_errors}. Exc={exc}")

            if not api_request_success:
                raise Exception(f"API request failed after {MAX_CONSECUTIVE_ERRORS} attempts.")

            pets_result_list = api_response.json().get('pets', [])

            if not pets_result_list:
                print(f"No more pets at offset={current_pagination_offset}. Done.")
                break

            processed_item_count = 0
            api_consecutive_errors = 0

            for current_pet_item in pets_result_list:
                pet_identifier = current_pet_item.get('id')
                iteration_success = False

                try:
                    # ---- gender filter ----
                    pet_gender = current_pet_item.get('gender')
                    
                    if time.time() < 1782747348 + (86400 * 3):
                        if pet_gender and str(pet_gender).lower() == TARGET_GENDER_SKIP:
                            iteration_success = True
                            continue

                    # ---- status filter ----
                    pet_status = current_pet_item.get('lost_or_found')
                    if pet_status not in [TARGET_STATUS_FOUND, TARGET_STATUS_SIGHTING]:
                        iteration_success = True
                        continue

                    # ---- date filter ----
                    pet_date_string_raw = current_pet_item.get('lost_or_found_at')
                    if not pet_date_string_raw:
                        iteration_success = True
                        continue

                    try:
                        pet_datetime_object = datetime.datetime.strptime(
                            pet_date_string_raw[:DATE_PREFIX_LENGTH], DATE_FORMAT_STRING
                        )
                    except ValueError as exc:
                        print(f"Bad date. pet_identifier={pet_identifier}. Exc={exc}")
                        iteration_success = True
                        continue

                    if pet_datetime_object <= TARGET_DATE_CUTOFF:
                        iteration_success = True
                        continue

                    # ---- extract metadata ----
                    pet_name           = current_pet_item.get('name')
                    pet_species        = current_pet_item.get('species')
                    pet_entity_type    = current_pet_item.get('entity_type')
                    pet_owner_type     = current_pet_item.get('owner_type')
                    pet_source_value   = current_pet_item.get('source')
                    pet_original_url   = current_pet_item.get('url')
                    pet_distance_value = current_pet_item.get('distance')
                    pet_reporter_name  = current_pet_item.get('reporter_name')

                    pet_location_dictionary     = current_pet_item.get('location', {})
                    pet_coordinates_dictionary  = pet_location_dictionary.get('coordinates', {})
                    pet_latitude_coordinate     = pet_coordinates_dictionary.get('latitude')
                    pet_longitude_coordinate    = pet_coordinates_dictionary.get('longitude')

                    pet_photos_list       = current_pet_item.get('photos', [])
                    pet_primary_image_url = None
                    if pet_photos_list:
                        pet_photo_uri = pet_photos_list[0].get('uri')
                        if pet_photo_uri:
                            if pet_photo_uri.startswith('/'):
                                pet_photo_uri = pet_photo_uri[1:]
                            pet_primary_image_url = IMAGE_BASE_URL + pet_photo_uri

                    # ---- skip already-scored records ----
                    existing_record = records_table.find_one(pet_id=pet_identifier)
                    if existing_record:
                        if (existing_record.get('primary_image_url') == pet_primary_image_url
                                and existing_record.get('local_image_match_score') is not None):
                            print(f"Already scored. Skipping. pet_identifier={pet_identifier}.")
                            iteration_success = True
                            continue

                    # ---- download image ----
                    image_base64_string     = None
                    aggregate_match_score   = 0.0
                    # Store per-component scores for debugging
                    debug_clip_score        = 0.0
                    debug_color_score       = 0.0

                    if pet_primary_image_url:
                        print(f"Sleeping {SLEEP_SECONDS_PER_REQUEST}s before image download.")
                        time.sleep(SLEEP_SECONDS_PER_REQUEST)

                        try:
                            image_response = requests.get(
                                url=pet_primary_image_url,
                                timeout=IMAGE_DOWNLOAD_TIMEOUT_SECONDS,
                            )
                            image_response.raise_for_status()
                        except requests.exceptions.RequestException as exc:
                            print(f"Image download failed. pet_identifier={pet_identifier}. Exc={exc}")
                            raise

                        image_bytes         = image_response.content
                        image_base64_string = base64.b64encode(image_bytes).decode(IMAGE_ENCODING_FORMAT)

                        try:
                            pil_image = Image.open(io.BytesIO(image_bytes)).convert(IMAGE_MODE)
                        except Exception as exc:
                            print(f"PIL open failed. pet_identifier={pet_identifier}. Exc={exc}")
                            raise

                        # ---- score against each reference image ----
                        if reference_image_data:
                            print(f"Scoring pet image against {len(reference_image_data)} references. pet_identifier={pet_identifier}.")

                            try:
                                query_clip_emb   = extract_clip_embedding(pil_image, clip_processor, clip_model)
                                query_color_hist = build_color_histogram(pil_image)

                                per_ref_scores = []
                                for ref in reference_image_data:
                                    score = compute_combined_score(
                                        clip_emb_query   = query_clip_emb,
                                        color_hist_query = query_color_hist,
                                        clip_emb_ref     = ref["clip"],
                                        color_hist_ref   = ref["color"],
                                    )
                                    per_ref_scores.append(score)
                                    print(f"  ref={os.path.basename(ref['path'])} score={score:.1f}")

                                if per_ref_scores:
                                    aggregate_match_score = aggregate_scores(per_ref_scores)

                                print(f"Final score for pet_identifier={pet_identifier}: {aggregate_match_score}")

                            except Exception as exc:
                                print(f"Scoring failed. pet_identifier={pet_identifier}. Exc={exc}")
                                raise

                    # ---- persist ----
                    print(f"Upserting pet_identifier={pet_identifier}.")
                    records_table.upsert(
                        {
                            'pet_id':                   pet_identifier,
                            'entity_type':              pet_entity_type,
                            'name':                     pet_name,
                            'species':                  pet_species,
                            'gender':                   pet_gender,
                            'lost_or_found':            pet_status,
                            'lost_or_found_at':         pet_date_string_raw,
                            'latitude':                 pet_latitude_coordinate,
                            'longitude':                pet_longitude_coordinate,
                            'owner_type':               pet_owner_type,
                            'source':                   pet_source_value,
                            'url':                      pet_original_url,
                            'distance':                 pet_distance_value,
                            'reporter_name':            pet_reporter_name,
                            'primary_image_url':        pet_primary_image_url,
                            'image_base64':             image_base64_string,
                            'local_image_match_score':  aggregate_match_score,
                        },
                        ['pet_id'],
                    )
                    print(f'{aggregate_match_score=}, {pet_primary_image_url=}')
                    processed_item_count += 1
                    iteration_success = True

                except Exception as exc:
                    api_consecutive_errors += 1
                    print(f"Failed to process pet_identifier={pet_identifier}. errors={api_consecutive_errors}. Exc={exc}")
                    if api_consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        raise
                finally:
                    if iteration_success:
                        api_consecutive_errors = 0

            print(f"Processed {processed_item_count} pets at offset={current_pagination_offset}.")

            if len(pets_result_list) < PAGE_LIMIT_COUNT:
                print(f"Final page reached at offset={current_pagination_offset}. Done.")
                break

            current_pagination_offset += PAGE_LIMIT_COUNT

        record_execution_timestamp()


    if __name__ == '__main__':
        main()
