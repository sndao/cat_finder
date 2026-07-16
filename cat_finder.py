# ~/cat_finder/cat_finder.py
# Scrapes Petco Love Lost found/sighting cat reports across a coverage grid, downloads every
# photo to ~/cat_finder/images/, scores it against local reference photos (CLIP + masked HSV
# histogram), and applies an auto-calibrated ORANGE COLOR GATE that zeroes out non-ginger cats.
# Likely matches are copied to ~/cat_finder/orange_cat_images/ as <score>_<pet_id>.<ext> so `ls`
# ranks them. All DB image paths are RELATIVE to ~/cat_finder so the repo can be cloned and the
# database rebuilt from ~/cat_finder/images/ alone. Home coords are hardcoded (Nominatim cannot
# resolve this cul-de-sac). Every row gets distance_from_home_miles for sorting.
# On startup: backfills legacy base64 rows, wipes stale paths, deletes photoless rows, rescores
# anything on a stale scoring_version. After the scrape: reduces all images to <=1280px JPEG.

import os
import io
import sys
import glob
import math
import time
import base64
import datetime

import requests
import dataset
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

from transformers import CLIPProcessor, CLIPModel

print(sys.executable)

# ============================================================
# Paths and Configuration
#
# CAT_FINDER_ROOT is the anchor. Everything stored in the DB is relative to it, so a fresh
# clone on another machine rebuilds without touching a single row.
# ============================================================
CAT_FINDER_ROOT = os.path.expanduser('~/cat_finder')

API_URL = 'https://api.lost2.petcolove.org/maps/pets-with-auth/search'
DATABASE_DIRECTORY = os.path.expanduser('~/data/databases')
DATABASE_FILE_PATH = os.path.join(DATABASE_DIRECTORY, 'found_cats.db')
STATE_FILE_PATH = os.path.join(DATABASE_DIRECTORY, 'last_run_timestamp.txt')
DATABASE_CONNECTION_URL = 'sqlite:///' + DATABASE_FILE_PATH
TABLE_NAME = 'found_cats'

REFERENCE_IMAGE_DIRECTORY = os.path.join(CAT_FINDER_ROOT, 'found_cat', 'images')
ALL_IMAGE_DIRECTORY_NAME = 'images'
ORANGE_IMAGE_DIRECTORY_NAME = 'orange_cat_images'
ALL_IMAGE_DIRECTORY = os.path.join(CAT_FINDER_ROOT, ALL_IMAGE_DIRECTORY_NAME)
ORANGE_IMAGE_DIRECTORY = os.path.join(CAT_FINDER_ROOT, ORANGE_IMAGE_DIRECTORY_NAME)

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

TARGET_DATE_CUTOFF = datetime.datetime(
    year=TARGET_DATE_YEAR,
    month=TARGET_DATE_MONTH,
    day=TARGET_DATE_DAY,
)

API_TOKEN_VALUE = '482f7a44f98ceec8b96ef4d72565fb2b'
API_SPECIES_VALUE = 'cat'
API_START_DATE_VALUE = '2026-04-01'

if time.time() > 1782749621 + (86400 * 10):
    API_START_DATE_VALUE = '2026-05-01'

if time.time() > 1782749621 + (86400 * 15):
    API_START_DATE_VALUE = '2026-06-01'

API_CREATED_AFTER_VALUE = '2026-04-06'

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

DEFAULT_IMAGE_EXTENSION = '.jpg'

# Bump this whenever scoring or storage logic changes -- stale rows get rebuilt on startup.
# v2: fixed the RGBA->RGB black-fill bug, added masked histograms + orange gate.
# v3: ORANGE_SATURATION_MIN 40->60, shortlist got its own admission threshold.
# v4: relative image paths, home coords geocoded from address (then hardcoded when Nominatim
#     failed), photoless rows deleted, distance column fixed (old origin was 20mi off).
# v5: reduce_image_sizes() runs after scrape; PNG->JPG renames require path rebuild.
SCORING_VERSION = 5

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

# ============================================================
# HOME LOCATION
#
# Hardcoded from Google Maps. Nominatim cannot resolve this address (new suburban
# cul-de-sac, not in OSM). The old hardcoded value (29.9446, -95.5099) was northwest
# Houston, ~22 miles off -- every distance_from_home_miles pre-v4 was wrong.
# ============================================================
HOME_ADDRESS = '18418 Pitmedden Court, Richmond, TX 77407'
HOME_LATITUDE = 29.62593582041919
HOME_LONGITUDE = -95.69813580000118

# ============================================================
# SEARCH GRID
# ============================================================
API_RADIUS_MILES = 100
GRID_OVERLAP_FACTOR = 0.95

GRID_LAT_MIN = 23.0
GRID_LAT_MAX = 37.0
GRID_LON_MIN = -107.0
GRID_LON_MAX = -88.0

MILES_PER_DEGREE_LATITUDE = 69.0
EARTH_RADIUS_MILES = 3958.8

# ============================================================
# GEOCODING
# ============================================================
GEOCODE_REVERSE_URL = 'https://nominatim.openstreetmap.org/reverse'
GEOCODE_USER_AGENT = 'cat_finder/1.0 (personal lost-pet search)'
GEOCODE_TIMEOUT_SECONDS = 10
GEOCODE_SLEEP_SECONDS = 1.1
GEOCODE_ZOOM = 10
GEOCODE_CACHE_PRECISION = 2

GEOCODE_CACHE = {}

# ============================================================
# SUBJECT MASKING
#
# ROOT CAUSE of the black-cat bug: PIL's .convert('RGB') on an RGBA image fills transparent
# pixels with BLACK. The background-removed reference PNGs were 42-77% pure black after that
# call. The HSV histogram encoded "this cat is mostly black" and matched black cats.
#
# REFERENCE IMAGES (have alpha): mask from alpha, composite onto neutral grey.
# QUERY IMAGES (real photos): center crop minus HSV bins that dominate the border (background).
# ============================================================
NEUTRAL_BACKGROUND_RGB = (128, 128, 128)
QUERY_CENTER_CROP_FRACTION = 0.72
BORDER_RING_FRACTION = 0.12
BACKGROUND_DOMINANCE_RATIO = 1.0
MIN_FOREGROUND_PIXEL_FRACTION = 0.03
HSV_QUANTIZATION_BIN_COUNT = 1152

# ============================================================
# ORANGE COLOR GATE
#
# SATURATION IS THE LOAD-BEARING PARAMETER. At SAT_MIN=40 a tuxedo/grey/black cat all
# measured 8.6-8.9% "orange" (brown shadow + JPEG chroma noise in hue band at low sat).
# Real ginger fur has median saturation 146-165. SAT_MIN=60 gives 4.6x margin.
#
#   SAT_MIN=40 -> ref_min 36.8%  vs negatives 8.7-8.9%  (no separation)
#   SAT_MIN=60 -> ref_min 20.7%  vs negatives 2.1-4.5%  (chosen)
#   SAT_MIN=80 -> ref_min  4.1%  (pale cream reference starts collapsing)
#
# Two thresholds, both auto-calibrated from the weakest reference:
#   GATE (ORANGE_GATE_RATIO): soft score multiplier. Generous.
#   SHORTLIST (ORANGE_DIR_RATIO): hard admission bar for orange_cat_images/. Stricter.
# ============================================================
ORANGE_HUE_MIN = 3
ORANGE_HUE_MAX = 30
ORANGE_SATURATION_MIN = 60
ORANGE_VALUE_MIN = 60

ORANGE_GATE_RATIO = 0.45
ORANGE_DIR_RATIO = 0.50
ORANGE_GATE_FLOOR = 0.08
ORANGE_GATE_EXPONENT = 2.0

ORANGE_DIR_MIN_SCORE = 45.0

# ============================================================
# SCORING WEIGHTS
# ============================================================
CLIP_WEIGHT = 0.55
COLOR_HIST_WEIGHT = 0.45

CLIP_SIMILARITY_FLOOR = 0.75
CLIP_SIMILARITY_CEIL = 0.99

SOFTMAX_TEMPERATURE = 0.10
BEST_OF_BLEND = 0.50

# ============================================================
# IMAGE REDUCTION
#
# Petco Love serves phone-camera PNGs at 3-8 MB. CLIP uses 224x224 and the HSV histogram
# doesn't care about resolution beyond ~720p. Re-encode to <=1280px JPEG at quality 88.
# Typical savings: 90-95%. Atomic writes (.tmp then os.replace). Idempotent.
#
# PNG->JPG renames the file, which makes the DB path stale. The rescore pass in
# migrate_and_rescore() wipes and rebuilds all path columns from disk, so running with
# SCORING_VERSION bumped after this is safe -- nothing is lost.
# ============================================================
MAX_LONG_EDGE_PIXELS = 1280
REENCODE_JPEG_QUALITY = 88
SKIP_IF_UNDER_BYTES = 250 * 1024


# ============================================================
# Path helpers -- DB stores relative, filesystem gets absolute
# ============================================================

def to_relative_path(absolute_path: str) -> str:
    """~/cat_finder/images/123.jpeg -> images/123.jpeg"""
    return os.path.relpath(absolute_path, CAT_FINDER_ROOT)


def to_absolute_path(relative_path: str) -> str:
    """images/123.jpeg -> ~/cat_finder/images/123.jpeg"""
    return os.path.join(CAT_FINDER_ROOT, relative_path)


def find_image_for_pet(pet_identifier) -> str:
    """
    Locate a pet's photo on disk by id, ignoring whatever the DB claims. Returns a RELATIVE
    path, or None. The filesystem is the source of truth; path columns are derived.
    """
    for candidate_path in glob.glob(os.path.join(ALL_IMAGE_DIRECTORY, f"{pet_identifier}.*")):
        if os.path.splitext(candidate_path)[1].lower() in VALID_IMAGE_EXTENSIONS:
            return to_relative_path(absolute_path=candidate_path)
    return None


# ============================================================
# Geo helpers
# ============================================================

def haversine_miles(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    """Great-circle distance in miles between two lat/lon points."""
    lat_a_rad = math.radians(lat_a)
    lat_b_rad = math.radians(lat_b)
    delta_lat = math.radians(lat_b - lat_a)
    delta_lon = math.radians(lon_b - lon_a)

    a = (math.sin(delta_lat / 2.0) ** 2) + (
        math.cos(lat_a_rad) * math.cos(lat_b_rad) * (math.sin(delta_lon / 2.0) ** 2)
    )
    return round(2.0 * EARTH_RADIUS_MILES * math.asin(math.sqrt(a)), 2)


def distance_from_home_miles(latitude, longitude):
    """None-safe haversine from HOME_LATITUDE/HOME_LONGITUDE."""
    if latitude is None or longitude is None:
        return None
    return haversine_miles(
        lat_a=HOME_LATITUDE,
        lon_a=HOME_LONGITUDE,
        lat_b=float(latitude),
        lon_b=float(longitude),
    )


def build_search_grid() -> list:
    """Square-packed coverage grid of (lat, lon) search centers, nearest-to-home first."""
    spacing_miles = API_RADIUS_MILES * math.sqrt(2.0) * GRID_OVERLAP_FACTOR
    latitude_step = spacing_miles / MILES_PER_DEGREE_LATITUDE

    grid_points = []
    current_latitude = GRID_LAT_MIN
    while current_latitude <= GRID_LAT_MAX + latitude_step:
        miles_per_degree_longitude = MILES_PER_DEGREE_LATITUDE * math.cos(math.radians(current_latitude))
        longitude_step = spacing_miles / max(miles_per_degree_longitude, 1.0)

        current_longitude = GRID_LON_MIN
        while current_longitude <= GRID_LON_MAX + longitude_step:
            grid_points.append((current_latitude, current_longitude))
            current_longitude += longitude_step

        current_latitude += latitude_step

    grid_points.sort(
        key=lambda point: haversine_miles(
            lat_a=HOME_LATITUDE,
            lon_a=HOME_LONGITUDE,
            lat_b=point[0],
            lon_b=point[1],
        )
    )
    return [(f"{lat:.8f}", f"{lon:.8f}") for lat, lon in grid_points]


def reverse_geocode(latitude, longitude) -> dict:
    """
    {'city','state','zip_code'} for a coordinate pair, cached by rounded coords.
    Failures are logged loudly but must never abort the scrape.
    """
    empty_result = {'city': None, 'state': None, 'zip_code': None}
    if latitude is None or longitude is None:
        return empty_result

    cache_key = (
        round(float(latitude), GEOCODE_CACHE_PRECISION),
        round(float(longitude), GEOCODE_CACHE_PRECISION),
    )
    if cache_key in GEOCODE_CACHE:
        return GEOCODE_CACHE[cache_key]

    try:
        time.sleep(GEOCODE_SLEEP_SECONDS)
        geocode_response = requests.get(
            url=GEOCODE_REVERSE_URL,
            params={
                'lat': str(latitude),
                'lon': str(longitude),
                'format': 'jsonv2',
                'zoom': str(GEOCODE_ZOOM),
                'addressdetails': '1',
            },
            headers={'user-agent': GEOCODE_USER_AGENT},
            timeout=GEOCODE_TIMEOUT_SECONDS,
        )
        geocode_response.raise_for_status()
        address_dictionary = geocode_response.json().get('address', {})

        result = {
            'city': (
                address_dictionary.get('city')
                or address_dictionary.get('town')
                or address_dictionary.get('village')
                or address_dictionary.get('hamlet')
                or address_dictionary.get('county')
            ),
            'state': address_dictionary.get('state'),
            'zip_code': address_dictionary.get('postcode'),
        }
    except Exception as exc:
        print(f"FAILED: reverse geocode. latitude={latitude}, longitude={longitude}. Exc={exc}", file=sys.stderr)
        result = empty_result

    GEOCODE_CACHE[cache_key] = result
    return result


# ============================================================
# Image loading, masking, color features
# ============================================================

def to_hsv_array(rgb_image: Image.Image) -> np.ndarray:
    """PIL RGB -> OpenCV HSV ndarray (H 0-179, S 0-255, V 0-255)."""
    return cv2.cvtColor(cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)


def load_image_and_mask(pil_image: Image.Image):
    """
    Return (rgb_image, subject_mask).

    Alpha present: mask from alpha, composite onto neutral grey.
    No alpha: center crop minus HSV bins that dominate the border (background).
    """
    if pil_image.mode in ('RGBA', 'LA', 'P'):
        rgba_image = pil_image.convert('RGBA')
        subject_mask = np.array(rgba_image.getchannel('A')) > 128
        backdrop = Image.new('RGBA', rgba_image.size, NEUTRAL_BACKGROUND_RGB + (255,))
        rgb_image = Image.alpha_composite(backdrop, rgba_image).convert(IMAGE_MODE)

        if not subject_mask.any():
            subject_mask = np.ones(subject_mask.shape, dtype=bool)
        return rgb_image, subject_mask

    rgb_image = pil_image.convert(IMAGE_MODE)
    hsv_array = to_hsv_array(rgb_image=rgb_image)
    image_height, image_width = hsv_array.shape[:2]

    quantized_bins = (
        (hsv_array[:, :, 0].astype(np.int32) // 10) * 64
        + (hsv_array[:, :, 1].astype(np.int32) // 32) * 8
        + (hsv_array[:, :, 2].astype(np.int32) // 32)
    )
    quantized_bins = np.clip(quantized_bins, 0, HSV_QUANTIZATION_BIN_COUNT - 1)

    center_mask = np.zeros((image_height, image_width), dtype=bool)
    crop_height = int(image_height * QUERY_CENTER_CROP_FRACTION)
    crop_width = int(image_width * QUERY_CENTER_CROP_FRACTION)
    crop_y = (image_height - crop_height) // 2
    crop_x = (image_width - crop_width) // 2
    center_mask[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width] = True

    border_mask = np.zeros((image_height, image_width), dtype=bool)
    border_y = max(int(image_height * BORDER_RING_FRACTION), 1)
    border_x = max(int(image_width * BORDER_RING_FRACTION), 1)
    border_mask[:border_y, :] = True
    border_mask[-border_y:, :] = True
    border_mask[:, :border_x] = True
    border_mask[:, -border_x:] = True

    center_histogram = np.bincount(quantized_bins[center_mask], minlength=HSV_QUANTIZATION_BIN_COUNT).astype(np.float64)
    border_histogram = np.bincount(quantized_bins[border_mask], minlength=HSV_QUANTIZATION_BIN_COUNT).astype(np.float64)
    center_histogram /= max(center_histogram.sum(), 1.0)
    border_histogram /= max(border_histogram.sum(), 1.0)

    foreground_bins = center_histogram > (border_histogram * BACKGROUND_DOMINANCE_RATIO)
    subject_mask = center_mask & foreground_bins[quantized_bins]

    if subject_mask.sum() < (image_height * image_width * MIN_FOREGROUND_PIXEL_FRACTION):
        subject_mask = center_mask

    return rgb_image, subject_mask


def compute_orange_fraction(rgb_image: Image.Image, subject_mask: np.ndarray) -> float:
    """Fraction of subject pixels that are genuinely orange/ginger/tan fur."""
    hsv_array = to_hsv_array(rgb_image=rgb_image)

    orange_pixels = (
        (hsv_array[:, :, 0] >= ORANGE_HUE_MIN)
        & (hsv_array[:, :, 0] <= ORANGE_HUE_MAX)
        & (hsv_array[:, :, 1] >= ORANGE_SATURATION_MIN)
        & (hsv_array[:, :, 2] >= ORANGE_VALUE_MIN)
        & subject_mask
    )

    subject_pixel_count = int(subject_mask.sum())
    if subject_pixel_count == 0:
        return 0.0
    return round(float(orange_pixels.sum()) / subject_pixel_count, 4)


def build_color_histogram(rgb_image: Image.Image, subject_mask: np.ndarray, bins: int = 32) -> np.ndarray:
    """Normalised HSV histogram over subject pixels only."""
    hsv_array = to_hsv_array(rgb_image=rgb_image)
    calc_hist_mask = subject_mask.astype(np.uint8) * 255

    hue_histogram = cv2.calcHist([hsv_array], [0], calc_hist_mask, [bins], [0, 180]).flatten()
    saturation_histogram = cv2.calcHist([hsv_array], [1], calc_hist_mask, [bins], [0, 256]).flatten()
    value_histogram = cv2.calcHist([hsv_array], [2], calc_hist_mask, [bins], [0, 256]).flatten()

    weighted_histogram = np.concatenate([
        hue_histogram * 0.60,
        saturation_histogram * 0.30,
        value_histogram * 0.10,
    ])

    histogram_total = weighted_histogram.sum()
    if histogram_total > 0:
        weighted_histogram /= histogram_total

    return weighted_histogram.astype(np.float32)


def color_histogram_similarity(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    """Bhattacharyya coefficient: 0.0 (no overlap) to 1.0 (identical)."""
    return float(np.sum(np.sqrt(hist_a * hist_b)))


# ============================================================
# CLIP
# ============================================================

def extract_clip_embedding(pil_image: Image.Image, processor, model) -> torch.Tensor:
    """CLIP embedding with no cropping -- CLIP's attention handles framing itself."""
    inputs = processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        raw = model.get_image_features(**inputs)

    if not isinstance(raw, torch.Tensor):
        if hasattr(raw, 'image_embeds'):
            raw = raw.image_embeds
        elif hasattr(raw, 'pooler_output'):
            raw = raw.pooler_output
        else:
            raw = raw[0]

    return F.normalize(raw, p=2, dim=1)


def clip_cosine_to_score(raw_cosine: float) -> float:
    """Map raw CLIP cosine -> [0,1] across [FLOOR, CEIL]."""
    clipped_cosine = max(CLIP_SIMILARITY_FLOOR, min(CLIP_SIMILARITY_CEIL, raw_cosine))
    similarity_span = CLIP_SIMILARITY_CEIL - CLIP_SIMILARITY_FLOOR
    if similarity_span == 0:
        return 1.0
    return (clipped_cosine - CLIP_SIMILARITY_FLOOR) / similarity_span


def compute_combined_score(clip_emb_query, color_hist_query, clip_emb_ref, color_hist_ref) -> float:
    """CLIP similarity + masked color histogram similarity -> 0-100 score (pre-gate)."""
    raw_cosine = torch.matmul(clip_emb_query, clip_emb_ref.T).item()
    clip_score_01 = clip_cosine_to_score(raw_cosine=raw_cosine)
    color_score_01 = color_histogram_similarity(hist_a=color_hist_query, hist_b=color_hist_ref)

    combined_01 = (CLIP_WEIGHT * clip_score_01) + (COLOR_HIST_WEIGHT * color_score_01)
    return round(combined_01 * 100.0, 2)


def aggregate_scores(per_ref_scores: list) -> float:
    """Softmax-weighted mean blended with the best single score."""
    if not per_ref_scores:
        return 0.0
    if len(per_ref_scores) == 1:
        return round(per_ref_scores[0], 2)

    scores_np = np.array(per_ref_scores, dtype=np.float64)

    shifted = (scores_np - scores_np.max()) / SOFTMAX_TEMPERATURE
    softmax_weights = np.exp(shifted)
    softmax_weights /= softmax_weights.sum()

    softmax_mean = float(np.dot(softmax_weights, scores_np))
    best_score = float(scores_np.max())

    blended = (1.0 - BEST_OF_BLEND) * softmax_mean + BEST_OF_BLEND * best_score
    return round(float(np.clip(blended, 0.0, 100.0)), 2)


def score_image_bytes(image_bytes, reference_image_data, orange_gate_threshold, clip_processor, clip_model, verbose=True):
    """
    Full scoring pipeline for one photo.
    Returns (final_score, raw_score, orange_fraction, orange_gate_multiplier).
    final_score = raw_score * gate_multiplier.
    """
    query_rgb_image, query_subject_mask = load_image_and_mask(pil_image=Image.open(io.BytesIO(image_bytes)))

    orange_fraction = compute_orange_fraction(rgb_image=query_rgb_image, subject_mask=query_subject_mask)
    orange_gate_multiplier = round(min(orange_fraction / orange_gate_threshold, 1.0) ** ORANGE_GATE_EXPONENT, 4)

    if not reference_image_data:
        return 0.0, 0.0, orange_fraction, orange_gate_multiplier

    query_clip_embedding = extract_clip_embedding(
        pil_image=query_rgb_image,
        processor=clip_processor,
        model=clip_model,
    )
    query_color_histogram = build_color_histogram(
        rgb_image=query_rgb_image,
        subject_mask=query_subject_mask,
    )

    per_ref_scores = []
    for reference_entry in reference_image_data:
        reference_score = compute_combined_score(
            clip_emb_query=query_clip_embedding,
            color_hist_query=query_color_histogram,
            clip_emb_ref=reference_entry["clip"],
            color_hist_ref=reference_entry["color"],
        )
        per_ref_scores.append(reference_score)
        if verbose:
            print(f"  ref={os.path.basename(reference_entry['path'])} score={reference_score:.1f}")

    raw_score = aggregate_scores(per_ref_scores=per_ref_scores)
    final_score = round(raw_score * orange_gate_multiplier, 2)

    return final_score, raw_score, orange_fraction, orange_gate_multiplier


# ============================================================
# Image persistence -- writes files, returns RELATIVE paths for the DB
# ============================================================

def save_image_bytes(image_bytes, pet_identifier, source_url) -> str:
    """Write raw bytes to ~/cat_finder/images/<pet_id>.<ext>. Returns a RELATIVE path."""
    os.makedirs(ALL_IMAGE_DIRECTORY, exist_ok=True)

    file_extension = ''
    if source_url:
        url_extension = os.path.splitext(source_url.split('?')[0])[1].lower()
        if url_extension in VALID_IMAGE_EXTENSIONS:
            file_extension = url_extension

    if not file_extension:
        detected_format = (Image.open(io.BytesIO(image_bytes)).format or '').lower()
        file_extension = f".{detected_format}" if detected_format else DEFAULT_IMAGE_EXTENSION
        if file_extension == '.jpeg':
            file_extension = '.jpg'

    absolute_image_path = os.path.join(ALL_IMAGE_DIRECTORY, f"{pet_identifier}{file_extension}")
    with open(absolute_image_path, 'wb') as fh:
        fh.write(image_bytes)

    return to_relative_path(absolute_path=absolute_image_path)


def promote_to_orange_directory(relative_image_path, pet_identifier, final_score, orange_fraction, orange_dir_threshold) -> str:
    """
    Copy likely matches into orange_cat_images/ as <score>_<pet_id>.<ext>.
    Returns a RELATIVE path, or None if not admitted. Idempotent across rescores.
    """
    passes_color = orange_fraction >= orange_dir_threshold
    passes_score = final_score >= ORANGE_DIR_MIN_SCORE
    if not (passes_color or passes_score):
        return None

    os.makedirs(ORANGE_IMAGE_DIRECTORY, exist_ok=True)
    absolute_image_path = to_absolute_path(relative_path=relative_image_path)
    file_extension = os.path.splitext(absolute_image_path)[1].lower()
    absolute_orange_path = os.path.join(ORANGE_IMAGE_DIRECTORY, f"{final_score:05.1f}_{pet_identifier}{file_extension}")

    for stale_path in glob.glob(os.path.join(ORANGE_IMAGE_DIRECTORY, f"*_{pet_identifier}.*")):
        if stale_path != absolute_orange_path:
            os.replace(stale_path, absolute_orange_path)

    if not os.path.exists(absolute_orange_path):
        with open(absolute_image_path, 'rb') as source_fh, open(absolute_orange_path, 'wb') as target_fh:
            target_fh.write(source_fh.read())

    return to_relative_path(absolute_path=absolute_orange_path)


# ============================================================
# Startup migration + rescore
# ============================================================

def migrate_and_rescore(database_client, records_table, reference_image_data, orange_gate_threshold, orange_dir_threshold, clip_processor, clip_model):
    """
    1. Ensure columns exist.
    2. Backfill any legacy base64 rows to image files.
    3. Wipe stale path columns (they held absolute /Users/... paths) and rebuild from disk.
    4. Delete rows with no image on disk.
    5. Rescore all stale rows from the saved files.
    """
    for column_name, example_value in [
        ('image_file_path', 'x'),
        ('orange_image_file_path', 'x'),
        ('orange_fraction', 1.0),
        ('orange_gate_multiplier', 1.0),
        ('raw_match_score', 1.0),
        ('local_image_match_score', 1.0),
        ('scoring_version', 1),
        ('city', 'x'),
        ('state', 'x'),
        ('zip_code', 'x'),
        ('distance_from_home_miles', 1.0),
    ]:
        if column_name not in records_table.columns:
            records_table.create_column_by_example(name=column_name, value=example_value)
            print(f"Created column. column_name={column_name}.")

    # ---- base64 -> files ----
    if 'image_base64' in records_table.columns:
        base64_rows = list(database_client.query(
            f"SELECT pet_id, image_base64, primary_image_url FROM {TABLE_NAME} "
            f"WHERE image_base64 IS NOT NULL AND image_base64 != ''"
        ))
        print(f"Backfill: {len(base64_rows)} rows still carrying base64 image data.")

        for base64_row in base64_rows:
            pet_identifier = base64_row['pet_id']
            relative_image_path = save_image_bytes(
                image_bytes=base64.b64decode(base64_row['image_base64']),
                pet_identifier=pet_identifier,
                source_url=base64_row.get('primary_image_url'),
            )
            records_table.update(
                {'pet_id': pet_identifier, 'image_file_path': relative_image_path, 'image_base64': ''},
                ['pet_id'],
            )
            print(f"Backfilled base64 -> {relative_image_path}")

        if base64_rows:
            print("Running VACUUM to reclaim disk.")
            database_client.query('VACUUM')
            print("VACUUM done.")

    # ---- wipe path columns: they may hold stale absolute paths or corrupt orange shortlist ----
    database_client.query(f"UPDATE {TABLE_NAME} SET image_file_path = '', orange_image_file_path = ''")
    print("Wiped image_file_path and orange_image_file_path. Rebuilding from disk.")

    # ---- delete rows with no photo on disk ----
    all_pet_rows = list(database_client.query(f"SELECT pet_id FROM {TABLE_NAME}"))
    photoless_pet_ids = [
        row['pet_id'] for row in all_pet_rows
        if find_image_for_pet(pet_identifier=row['pet_id']) is None
    ]
    print(f"Found {len(photoless_pet_ids)} rows with no image on disk. Deleting.")
    for photoless_pet_id in photoless_pet_ids:
        records_table.delete(pet_id=photoless_pet_id)

    # ---- rescore stale rows ----
    stale_rows = list(database_client.query(
        f"SELECT pet_id, latitude, longitude, city FROM {TABLE_NAME} "
        f"WHERE scoring_version IS NULL OR scoring_version < {SCORING_VERSION}"
    ))
    print(f"Rescore: {len(stale_rows)} rows on a stale scoring_version.")

    for stale_row in stale_rows:
        pet_identifier = stale_row['pet_id']
        relative_image_path = find_image_for_pet(pet_identifier=pet_identifier)

        if relative_image_path is None:
            print(f"FAILED: image vanished mid-run. pet_identifier={pet_identifier}", file=sys.stderr)
            continue

        with open(to_absolute_path(relative_path=relative_image_path), 'rb') as fh:
            image_bytes = fh.read()

        final_score, raw_score, orange_fraction, orange_gate_multiplier = score_image_bytes(
            image_bytes=image_bytes,
            reference_image_data=reference_image_data,
            orange_gate_threshold=orange_gate_threshold,
            clip_processor=clip_processor,
            clip_model=clip_model,
            verbose=False,
        )

        relative_orange_path = promote_to_orange_directory(
            relative_image_path=relative_image_path,
            pet_identifier=pet_identifier,
            final_score=final_score,
            orange_fraction=orange_fraction,
            orange_dir_threshold=orange_dir_threshold,
        )

        update_payload = {
            'pet_id': pet_identifier,
            'image_file_path': relative_image_path,
            'orange_image_file_path': relative_orange_path,
            'local_image_match_score': final_score,
            'raw_match_score': raw_score,
            'orange_fraction': orange_fraction,
            'orange_gate_multiplier': orange_gate_multiplier,
            'distance_from_home_miles': distance_from_home_miles(
                latitude=stale_row.get('latitude'),
                longitude=stale_row.get('longitude'),
            ),
            'scoring_version': SCORING_VERSION,
        }

        if not stale_row.get('city'):
            update_payload.update(reverse_geocode(
                latitude=stale_row.get('latitude'),
                longitude=stale_row.get('longitude'),
            ))

        records_table.update(update_payload, ['pet_id'])
        print(f"Rebuilt pet_identifier={pet_identifier}: score={final_score} (raw={raw_score}, "
              f"orange={orange_fraction:.3f}) miles={update_payload['distance_from_home_miles']} "
              f"shortlisted={bool(relative_orange_path)}")

    print("Migration and rescore complete.")


# ============================================================
# State helpers
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
    os.makedirs(ALL_IMAGE_DIRECTORY, exist_ok=True)
    os.makedirs(ORANGE_IMAGE_DIRECTORY, exist_ok=True)

    print(f"Home: {HOME_ADDRESS} -> {HOME_LATITUDE}, {HOME_LONGITUDE}")

    database_client = dataset.connect(DATABASE_CONNECTION_URL)
    records_table = database_client[TABLE_NAME]

    # 1. Load CLIP once.
    print("Initializing CLIP model.")
    clip_processor = CLIPProcessor.from_pretrained(MODEL_ID)
    clip_model = CLIPModel.from_pretrained(MODEL_ID)

    # 2. Load reference photos, masked via alpha channel.
    reference_image_data = []
    reference_image_paths = glob.glob(os.path.join(REFERENCE_IMAGE_DIRECTORY, '*'))
    print(f"Scanning {len(reference_image_paths)} local reference files.")

    local_consecutive_errors = 0
    for reference_path in reference_image_paths:
        iteration_success = False
        file_extension = os.path.splitext(reference_path)[1].lower()
        if file_extension not in VALID_IMAGE_EXTENSIONS:
            print(f"Skipping non-image file. path={reference_path}.")
            continue

        try:
            reference_rgb, reference_mask = load_image_and_mask(pil_image=Image.open(reference_path))
            reference_orange_fraction = compute_orange_fraction(
                rgb_image=reference_rgb,
                subject_mask=reference_mask,
            )

            reference_image_data.append({
                "clip": extract_clip_embedding(
                    pil_image=reference_rgb,
                    processor=clip_processor,
                    model=clip_model,
                ),
                "color": build_color_histogram(rgb_image=reference_rgb, subject_mask=reference_mask),
                "orange": reference_orange_fraction,
                "path": reference_path,
            })

            print(f"Loaded reference. path={reference_path}, orange_fraction={reference_orange_fraction:.3f}, "
                  f"subject_pixels={int(reference_mask.sum())}")
            iteration_success = True

        except Exception as exc:
            local_consecutive_errors += 1
            print(f"FAILED: local reference image. path={reference_path}, errors={local_consecutive_errors}. Exc={exc}", file=sys.stderr)
            if local_consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                raise
        finally:
            if iteration_success:
                local_consecutive_errors = 0

    if not reference_image_data:
        raise Exception(f"No reference images loaded from {REFERENCE_IMAGE_DIRECTORY}. Cannot score anything.")

    # 3. Auto-calibrate both orange thresholds from the weakest reference.
    reference_orange_fractions = np.array([entry["orange"] for entry in reference_image_data])
    weakest_reference_orange = float(reference_orange_fractions.min())
    orange_gate_threshold = max(weakest_reference_orange * ORANGE_GATE_RATIO, ORANGE_GATE_FLOOR)
    orange_dir_threshold = max(weakest_reference_orange * ORANGE_DIR_RATIO, ORANGE_GATE_FLOOR)

    print(f"Loaded {len(reference_image_data)} references. orange_fraction "
          f"min={reference_orange_fractions.min():.3f} mean={reference_orange_fractions.mean():.3f} "
          f"max={reference_orange_fractions.max():.3f}")
    print(f"ORANGE GATE threshold={orange_gate_threshold:.4f} (score suppression)")
    print(f"SHORTLIST threshold={orange_dir_threshold:.4f} (admission to {ORANGE_IMAGE_DIRECTORY})")

    # 4. Migrate, rebuild paths, delete photoless rows, rescore stale.
    migrate_and_rescore(
        database_client=database_client,
        records_table=records_table,
        reference_image_data=reference_image_data,
        orange_gate_threshold=orange_gate_threshold,
        orange_dir_threshold=orange_dir_threshold,
        clip_processor=clip_processor,
        clip_model=clip_model,
    )

    # 5. Walk the coverage grid, nearest-to-home first.
    search_grid = build_search_grid()
    print(f"Search grid: {len(search_grid)} centers at radius={API_RADIUS_MILES} miles (old grid was 266).")

    seen_pet_ids = set()

    for grid_index, (grid_latitude, grid_longitude) in enumerate(search_grid):
        grid_distance = distance_from_home_miles(latitude=grid_latitude, longitude=grid_longitude)
        print(f"=== Grid {grid_index + 1}/{len(search_grid)}: latitude={grid_latitude}, "
              f"longitude={grid_longitude}, distance_from_home_miles={grid_distance} ===")

        base_request_params = {
            'token': API_TOKEN_VALUE,
            'species': API_SPECIES_VALUE,
            'start': API_START_DATE_VALUE,
            'types': [
                'foundUserPet',
                'foundOrgPet',
                'sighting',
            ],
            'created_after': API_CREATED_AFTER_VALUE,
            'radius': str(API_RADIUS_MILES),
            'latitude': grid_latitude,
            'longitude': grid_longitude,
            'limit': str(PAGE_LIMIT_COUNT),
        }

        current_pagination_offset = PAGINATION_START_OFFSET

        while True:
            request_parameters = base_request_params.copy()
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
                    print(f"FAILED: API request. errors={api_request_errors}. Exc={exc}", file=sys.stderr)

            if not api_request_success:
                raise Exception(f"API request failed after {MAX_CONSECUTIVE_ERRORS} attempts.")

            pets_result_list = api_response.json().get('pets', [])

            if not pets_result_list:
                print(f"No more pets at offset={current_pagination_offset}. Done with this grid center.")
                break

            processed_item_count = 0
            api_consecutive_errors = 0

            for current_pet_item in pets_result_list:
                pet_identifier = current_pet_item.get('id')
                iteration_success = False

                try:
                    # ---- already handled this run (overlapping circles) ----
                    if pet_identifier in seen_pet_ids:
                        iteration_success = True
                        continue

                    # ---- gender filter ----
                    pet_gender = current_pet_item.get('gender')
                    if time.time() < 1782747348 + (86400 * 3):
                        if pet_gender and str(pet_gender).lower() == TARGET_GENDER_SKIP:
                            seen_pet_ids.add(pet_identifier)
                            iteration_success = True
                            continue

                    # ---- status filter ----
                    pet_status = current_pet_item.get('lost_or_found')
                    # if pet_status not in [TARGET_STATUS_FOUND, TARGET_STATUS_SIGHTING]:
                    if pet_status in ['lost']:
                        seen_pet_ids.add(pet_identifier)
                        iteration_success = True
                        continue

                    # ---- date filter ----
                    pet_date_string_raw = current_pet_item.get('lost_or_found_at')
                    if not pet_date_string_raw:
                        seen_pet_ids.add(pet_identifier)
                        iteration_success = True
                        continue

                    try:
                        pet_datetime_object = datetime.datetime.strptime(
                            pet_date_string_raw[:DATE_PREFIX_LENGTH], DATE_FORMAT_STRING
                        )
                    except ValueError as exc:
                        print(f"FAILED: bad date. pet_identifier={pet_identifier}. Exc={exc}", file=sys.stderr)
                        seen_pet_ids.add(pet_identifier)
                        iteration_success = True
                        continue

                    if pet_datetime_object <= TARGET_DATE_CUTOFF:
                        seen_pet_ids.add(pet_identifier)
                        iteration_success = True
                        continue

                    # ---- no photo = no row ----
                    pet_photos_list = current_pet_item.get('photos', [])
                    pet_primary_image_url = None
                    if pet_photos_list:
                        pet_photo_uri = pet_photos_list[0].get('uri')
                        if pet_photo_uri:
                            if pet_photo_uri.startswith('/'):
                                pet_photo_uri = pet_photo_uri[1:]
                            pet_primary_image_url = IMAGE_BASE_URL + pet_photo_uri

                    if not pet_primary_image_url:
                        print(f"No photo. Skipping pet_identifier={pet_identifier}.")
                        seen_pet_ids.add(pet_identifier)
                        iteration_success = True
                        continue

                    # ---- metadata ----
                    pet_name = current_pet_item.get('name')
                    pet_species = current_pet_item.get('species')
                    pet_entity_type = current_pet_item.get('entity_type')
                    pet_owner_type = current_pet_item.get('owner_type')
                    pet_source_value = current_pet_item.get('source')
                    pet_original_url = current_pet_item.get('url')
                    pet_distance_value = current_pet_item.get('distance')
                    pet_reporter_name = current_pet_item.get('reporter_name')

                    pet_location_dictionary = current_pet_item.get('location', {})
                    pet_coordinates_dictionary = pet_location_dictionary.get('coordinates', {})
                    pet_latitude_coordinate = pet_coordinates_dictionary.get('latitude')
                    pet_longitude_coordinate = pet_coordinates_dictionary.get('longitude')

                    pet_distance_from_home = distance_from_home_miles(
                        latitude=pet_latitude_coordinate,
                        longitude=pet_longitude_coordinate,
                    )
                    location_details = reverse_geocode(
                        latitude=pet_latitude_coordinate,
                        longitude=pet_longitude_coordinate,
                    )

                    # ---- skip already-scored records ----
                    existing_record = records_table.find_one(pet_id=pet_identifier)
                    # if existing_record:
                    #     if (existing_record.get('primary_image_url') == pet_primary_image_url
                    #             and existing_record.get('scoring_version') == SCORING_VERSION):
                    #         print(f"Already scored. Skipping. pet_identifier={pet_identifier}.")
                    #         seen_pet_ids.add(pet_identifier)
                    #         iteration_success = True
                    #         continue

                    # ---- download and score ----
                    print(f"Sleeping {SLEEP_SECONDS_PER_REQUEST}s before image download.")
                    time.sleep(SLEEP_SECONDS_PER_REQUEST)

                    try:
                        image_response = requests.get(
                            url=pet_primary_image_url,
                            timeout=IMAGE_DOWNLOAD_TIMEOUT_SECONDS,
                        )
                        image_response.raise_for_status()
                    except requests.exceptions.RequestException as exc:
                        print(f"FAILED: image download. pet_identifier={pet_identifier}. Exc={exc}", file=sys.stderr)
                        raise

                    image_bytes = image_response.content

                    relative_image_path = save_image_bytes(
                        image_bytes=image_bytes,
                        pet_identifier=pet_identifier,
                        source_url=pet_primary_image_url,
                    )
                    print(f"Saved image. pet_identifier={pet_identifier}, image_file_path={relative_image_path}.")

                    try:
                        final_match_score, raw_match_score, orange_fraction, orange_gate_multiplier = score_image_bytes(
                            image_bytes=image_bytes,
                            reference_image_data=reference_image_data,
                            orange_gate_threshold=orange_gate_threshold,
                            clip_processor=clip_processor,
                            clip_model=clip_model,
                            verbose=True,
                        )
                    except Exception as exc:
                        print(f"FAILED: scoring. pet_identifier={pet_identifier}. Exc={exc}", file=sys.stderr)
                        raise

                    relative_orange_path = promote_to_orange_directory(
                        relative_image_path=relative_image_path,
                        pet_identifier=pet_identifier,
                        final_score=final_match_score,
                        orange_fraction=orange_fraction,
                        orange_dir_threshold=orange_dir_threshold,
                    )

                    print(f"pet_identifier={pet_identifier} score={final_match_score} (raw={raw_match_score}, "
                          f"orange_fraction={orange_fraction:.3f}, gate={orange_gate_multiplier:.3f}) "
                          f"miles={pet_distance_from_home} shortlisted={bool(relative_orange_path)}")

                    # ---- persist ----
                    records_table.upsert(
                        {
                            'pet_id': pet_identifier,
                            'entity_type': pet_entity_type,
                            'name': pet_name,
                            'species': pet_species,
                            'gender': pet_gender,
                            'lost_or_found': pet_status,
                            'lost_or_found_at': pet_date_string_raw,
                            'latitude': pet_latitude_coordinate,
                            'longitude': pet_longitude_coordinate,
                            'city': location_details['city'],
                            'state': location_details['state'],
                            'zip_code': location_details['zip_code'],
                            'distance_from_home_miles': pet_distance_from_home,
                            'owner_type': pet_owner_type,
                            'source': pet_source_value,
                            'url': pet_original_url,
                            'distance': pet_distance_value,
                            'reporter_name': pet_reporter_name,
                            'primary_image_url': pet_primary_image_url,
                            'image_file_path': relative_image_path,
                            'orange_image_file_path': relative_orange_path,
                            'image_base64': '',
                            'orange_fraction': orange_fraction,
                            'orange_gate_multiplier': orange_gate_multiplier,
                            'raw_match_score': raw_match_score,
                            'local_image_match_score': final_match_score,
                            'scoring_version': SCORING_VERSION,
                        },
                        ['pet_id'],
                    )

                    seen_pet_ids.add(pet_identifier)
                    processed_item_count += 1
                    iteration_success = True

                except Exception as exc:
                    api_consecutive_errors += 1
                    print(f"FAILED: process pet. pet_identifier={pet_identifier}, errors={api_consecutive_errors}. Exc={exc}", file=sys.stderr)
                    if api_consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        raise
                finally:
                    if iteration_success:
                        api_consecutive_errors = 0

            print(f"Processed {processed_item_count} pets at offset={current_pagination_offset}.")

            if len(pets_result_list) < PAGE_LIMIT_COUNT:
                print(f"Final page reached at offset={current_pagination_offset}. Done with this grid center.")
                break

            current_pagination_offset += PAGE_LIMIT_COUNT

    shortlist_count = len(glob.glob(os.path.join(ORANGE_IMAGE_DIRECTORY, '*')))
    print(f"Scrape complete. unique_pets_seen={len(seen_pet_ids)}, shortlist_size={shortlist_count}")
    record_execution_timestamp()

    # 6. Reduce images to <=1280px JPEG now that all downloads are done. PNG->JPG renames
    #    leave path columns stale; the next run's migrate_and_rescore() rebuilds them from
    #    disk. Scores are unaffected (CLIP uses 224x224; HSV histogram is resolution-invariant).
    reduce_image_sizes()


# ============================================================
# Image reduction
#
# Resize to MAX_LONG_EDGE_PIXELS and re-encode as JPEG quality 88. Atomic (.tmp then
# os.replace). Idempotent: already-small JPEGs are skipped. PNG->JPG removes the original
# PNG so pet_id globs in find_image_for_pet() still return exactly one file.
# ============================================================

def reduce_image_sizes(image_directory=ALL_IMAGE_DIRECTORY):
    """Downscale + re-encode every image in image_directory to a <=1280px JPEG."""
    candidate_paths = sorted(glob.glob(os.path.join(image_directory, '*')))
    print(f"reduce_image_sizes: scanning {len(candidate_paths)} files in {image_directory}.")

    total_bytes_before = 0
    total_bytes_after = 0
    reduced_count = 0
    skipped_count = 0

    for current_path in candidate_paths:
        file_extension = os.path.splitext(current_path)[1].lower()
        if file_extension not in VALID_IMAGE_EXTENSIONS:
            continue

        bytes_before = os.path.getsize(current_path)
        pil_image = Image.open(current_path)
        long_edge = max(pil_image.size)
        already_jpeg = file_extension in ('.jpg', '.jpeg')
        already_small = already_jpeg and (bytes_before < SKIP_IF_UNDER_BYTES) and (long_edge <= MAX_LONG_EDGE_PIXELS)

        if already_small:
            total_bytes_before += bytes_before
            total_bytes_after += bytes_before
            skipped_count += 1
            continue

        # Flatten alpha onto white -- JPEG has no alpha, and .convert('RGB') on an RGBA image
        # fills transparency with BLACK (the same bug that caused the black-cat scoring mess).
        if pil_image.mode in ('RGBA', 'LA', 'P'):
            rgba_image = pil_image.convert('RGBA')
            backdrop = Image.new('RGBA', rgba_image.size, (255, 255, 255, 255))
            pil_image = Image.alpha_composite(backdrop, rgba_image).convert('RGB')
        else:
            pil_image = pil_image.convert('RGB')

        if long_edge > MAX_LONG_EDGE_PIXELS:
            scale_factor = MAX_LONG_EDGE_PIXELS / float(long_edge)
            new_size = (
                max(int(pil_image.size[0] * scale_factor), 1),
                max(int(pil_image.size[1] * scale_factor), 1),
            )
            pil_image = pil_image.resize(new_size, Image.LANCZOS)

        target_path = os.path.splitext(current_path)[0] + '.jpg'
        temporary_path = target_path + '.tmp'

        pil_image.save(
            temporary_path,
            format='JPEG',
            quality=REENCODE_JPEG_QUALITY,
            optimize=True,
            progressive=True,
        )

        bytes_after = os.path.getsize(temporary_path)

        # If re-encoding an already-small JPEG would inflate it, keep the original.
        if bytes_after >= bytes_before and already_jpeg:
            os.remove(temporary_path)
            total_bytes_before += bytes_before
            total_bytes_after += bytes_before
            skipped_count += 1
            continue

        os.replace(temporary_path, target_path)

        if target_path != current_path:
            os.remove(current_path)

        total_bytes_before += bytes_before
        total_bytes_after += bytes_after
        reduced_count += 1

        print(f"  {os.path.basename(current_path)}: {bytes_before / 1e6:.2f} MB -> "
              f"{bytes_after / 1e6:.2f} MB ({pil_image.size[0]}x{pil_image.size[1]}px) "
              f"-> {os.path.basename(target_path)}")

    saved_bytes = total_bytes_before - total_bytes_after
    saved_percent = (saved_bytes / total_bytes_before * 100.0) if total_bytes_before else 0.0
    print(f"reduce_image_sizes: reduced={reduced_count}, skipped={skipped_count}, "
          f"before={total_bytes_before / 1e9:.2f} GB, after={total_bytes_after / 1e9:.2f} GB, "
          f"saved={saved_bytes / 1e9:.2f} GB ({saved_percent:.1f}%)")


if __name__ == '__main__':
    main()

# ~/cat_finder/venv/bin/python ~/cat_finder/cat_finder.py    # run