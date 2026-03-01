import os
import io
import datetime
from pymongo import MongoClient
import gridfs

try:
    import cloudinary
    import cloudinary.uploader
except Exception:
    cloudinary = None

# MongoDB connection string (override with MONGO_URI env var if needed)
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB", "rotten_or_not")

# Cloudinary defaults (override with env vars when deploying)
CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")


def get_db(uri: str = None, db_name: str = None):
    uri = uri or MONGO_URI
    db_name = db_name or DB_NAME
    client = MongoClient(uri, serverSelectionTimeoutMS=2000)
    db = client[db_name]
    return db


def upload_to_cloudinary(raw_bytes: bytes, filename: str, cloud_name: str = None, api_key: str = None, api_secret: str = None) -> dict:
    """Upload raw image bytes to Cloudinary and return the upload result dict."""
    if cloudinary is None:
        raise RuntimeError("cloudinary package is not installed")

    cloud_name = cloud_name or CLOUDINARY_CLOUD_NAME
    api_key = api_key or CLOUDINARY_API_KEY
    api_secret = api_secret or CLOUDINARY_API_SECRET

    if not (cloud_name and api_key and api_secret):
        raise ValueError("Cloudinary credentials are not provided")

    cloudinary.config(cloud_name=cloud_name, api_key=api_key, api_secret=api_secret)

    fp = io.BytesIO(raw_bytes)
    fp.name = filename
    # upload as streamed file; resource_type 'image' by default
    res = cloudinary.uploader.upload(fp, resource_type="image")
    return res


def save_upload(raw_bytes: bytes, filename: str, chosen_fruit: str, detected_info: object, uri: str = None, db_name: str = None, cloudinary_config: dict = None) -> dict:
    """Save an uploaded image and metadata.

    Behavior:
    - If `cloudinary_config` is provided (or CLOUDINARY_* env vars exist), upload the image to Cloudinary and store the Cloudinary response in metadata.
    - Otherwise, store the raw image bytes in GridFS and record the file id.

    Returns the metadata document inserted into MongoDB.
    """
    db = get_db(uri=uri, db_name=db_name)

    cloud_info = None
    if cloudinary_config is None:
        # try environment
        if CLOUDINARY_CLOUD_NAME and CLOUDINARY_API_KEY and CLOUDINARY_API_SECRET:
            cloudinary_config = {
                "cloud_name": CLOUDINARY_CLOUD_NAME,
                "api_key": CLOUDINARY_API_KEY,
                "api_secret": CLOUDINARY_API_SECRET,
            }

    if cloudinary_config:
        try:
            cloud_info = upload_to_cloudinary(raw_bytes, filename,
                                              cloud_name=cloudinary_config.get("cloud_name"),
                                              api_key=cloudinary_config.get("api_key"),
                                              api_secret=cloudinary_config.get("api_secret"))
        except Exception:
            cloud_info = None

    file_id = None
    if cloud_info is None:
        # fallback to GridFS storage
        fs = gridfs.GridFS(db)
        file_id = fs.put(raw_bytes, filename=filename)

    meta = {
        "filename": filename,
        "file_id": file_id,
        "cloudinary": cloud_info,
        "chosen_fruit": chosen_fruit,
        "detected_info": detected_info,
        "uploaded_at": datetime.datetime.utcnow(),
    }

    res = db.uploads.insert_one(meta)
    meta["_id"] = res.inserted_id
    return meta
