import os
import json
import base64
import tempfile
import threading
from typing import List, Dict, Any, Union, Optional
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

CHROMA_3D_LOAD_LOCK = threading.Lock()  # guards parallel calls during first-load


def get_gemini_embedding_client():
    """
    Returns Google GenAI client or Vertex AI fallback for gemini-embedding-2.
    """
    if "GEMINI_API_KEY" in os.environ:
        from google import genai
        return ("genai", genai.Client(api_key=os.getenv("GEMINI_API_KEY")))
    elif "GOOGLE_API_KEY" in os.environ:
        from google import genai
        return ("genai", genai.Client(api_key=os.getenv("GOOGLE_API_KEY")))
    elif "VERTEX_AI_SA_B64" in os.environ:
        key_bytes = base64.b64decode(os.environ["VERTEX_AI_SA_B64"])
        sa_dict = json.loads(key_bytes.decode())
        from google.oauth2 import service_account
        import vertexai
        from vertexai.vision_models import MultiModalEmbeddingModel

        creds = service_account.Credentials.from_service_account_info(sa_dict)
        vertexai.init(
            project=sa_dict.get("project_id", "dam-images-tagging"),
            location="us-central1",
            credentials=creds,
        )
        return ("vertex", MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001"))
    else:
        raise RuntimeError("No GEMINI_API_KEY, GOOGLE_API_KEY, or VERTEX_AI_SA_B64 found in environment.")


def embed_text_gemini(text: str) -> List[float]:
    """
    Generate vector for text query using gemini-embedding-2.
    """
    client_type, client = get_gemini_embedding_client()
    if client_type == "genai":
        from google.genai import types
        response = client.models.embed_content(
            model="gemini-embedding-2",
            contents=text,
        )
        return response.embeddings[0].values
    else:
        result = client.get_embeddings(contextual_text=text, dimension=1408)
        return result.text_embedding


def embed_image_gemini(image_input: Union[str, bytes, Image.Image]) -> List[float]:
    """
    Generate vector for image query using gemini-embedding-2 multimodal.
    """
    client_type, client = get_gemini_embedding_client()
    temp_path = None
    try:
        if isinstance(image_input, Image.Image):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                temp_path = tmp.name
                image_input.save(tmp.name, format="PNG")
        elif isinstance(image_input, bytes):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                temp_path = tmp.name
                tmp.write(image_input)
        elif isinstance(image_input, str) and os.path.exists(image_input):
            temp_path = image_input
        else:
            raise ValueError(f"Invalid image input for embedding: {type(image_input)}")

        if client_type == "genai":
            from google.genai import types
            with open(temp_path, "rb") as f:
                img_bytes = f.read()
            part = types.Part.from_bytes(data=img_bytes, mime_type="image/png")
            response = client.models.embed_content(
                model="gemini-embedding-2",
                contents=[part],
            )
            return response.embeddings[0].values
        else:
            from vertexai.vision_models import Image as VertexImage
            v_img = VertexImage.load_from_file(temp_path)
            result = client.get_embeddings(image=v_img, dimension=1408)
            return result.image_embedding
    finally:
        if temp_path and temp_path != image_input and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass


def embed_paired_gemini(
    text: str,
    media_bytes: Optional[bytes] = None,
    mime_type: Optional[str] = None,
) -> List[float]:
    """
    Gemini Embedding Aggregation (gemini-embedding-2 / text-embedding-004):
    Aggregates Cleaned Model text and Preview Media (image or video bytes) into a single unified vector.
    Ref: https://ai.google.dev/gemini-api/docs/embeddings#embedding-aggregation
    Ref: https://ai.google.dev/gemini-api/docs/embeddings#embed-video
    """
    client_type, client = get_gemini_embedding_client()

    if not media_bytes or not mime_type:
        return embed_text_gemini(text)

    if client_type == "genai":
        from google.genai import types

        contents_list = []
        if text and str(text).strip():
            contents_list.append(types.Part.from_text(text=str(text).strip()))

        # Add image or video bytes part
        contents_list.append(types.Part.from_bytes(data=media_bytes, mime_type=mime_type))

        response = client.models.embed_content(
            model="gemini-embedding-2",
            contents=contents_list,
        )
        return response.embeddings[0].values
    else:
        # Vertex AI fallback
        from vertexai.vision_models import Image as VertexImage
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
            tmp.write(media_bytes)
            tmp_path = tmp.name

        try:
            if "image" in mime_type:
                v_img = VertexImage.load_from_file(tmp_path)
                result = client.get_embeddings(image=v_img, contextual_text=text, dimension=1408)
                return result.image_embedding
            else:
                result = client.get_embeddings(contextual_text=text, dimension=1408)
                return result.text_embedding
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


def make_path_writable(path: str):
    """Recursively ensures directory and files have full read/write permissions for SQLite."""
    if not os.path.exists(path):
        return
    try:
        os.chmod(path, 0o777)
    except Exception:
        pass
    for root, dirs, files in os.walk(path):
        for d in dirs:
            try:
                os.chmod(os.path.join(root, d), 0o777)
            except Exception:
                pass
        for f in files:
            try:
                os.chmod(os.path.join(root, f), 0o666)
            except Exception:
                pass


def load_3d_chroma_db(drive=None, parent_folder_id="root", force_reload=False) -> Dict[str, Any]:
    """
    Loads the 3D models ChromaDB collection.
    Rule: if the local sqlite exists → use it. If not → download from Drive.
    force_reload=True deletes the local copy first so Drive is always re-downloaded.
    """
    import shutil
    from langchain_chroma import Chroma
    from services.drive_service import download_folder_from_drive

    local_chroma_root = "/tmp/temp_chroma_folder"
    local_chroma_path = os.path.join(local_chroma_root, f"chroma_3d_models_db_{parent_folder_id}")
    local_sqlite = os.path.join(local_chroma_path, "chroma.sqlite3")

    with CHROMA_3D_LOAD_LOCK:
        # force_reload → wipe local so Drive re-downloads unconditionally
        if force_reload and os.path.exists(local_chroma_path):
            shutil.rmtree(local_chroma_path, ignore_errors=True)
            print("🗑️ Wiped local ChromaDB (force_reload=True)")

        os.makedirs(local_chroma_path, exist_ok=True)

        # Simple rule: no local sqlite → download from Drive
        def _download_from_drive():
            if not drive:
                print("⚠️ No Drive credentials — starting with empty local DB.")
                return
            try:
                vstore_list = drive.ListFile({
                    'q': f"title='Vectorstore files' and '{parent_folder_id}' in parents "
                         f"and mimeType='application/vnd.google-apps.folder' and trashed=false"
                }).GetList()
                if vstore_list:
                    # Files are uploaded directly into Vectorstore files/ (no chroma_3d_models_db subfolder)
                    download_folder_from_drive(vstore_list[0]['id'], local_chroma_path, drive)
                    print("✅ Drive download complete.")
                else:
                    print("⚠️ Vectorstore files folder not found on Drive — starting empty.")
            except Exception as e:
                print(f"⚠️ Drive download failed: {e}")

        if not os.path.exists(local_sqlite):
            print("⬇️ No local DB found — downloading from Google Drive...")
            _download_from_drive()

        make_path_writable(local_chroma_path)

        unified_chroma = Chroma(
            collection_name="3d_models_unified",
            persist_directory=local_chroma_path,
        )

        print(f"📦 ChromaDB loaded — {unified_chroma._collection.count()} documents.")

        return {
            "unified": unified_chroma,
            "text": unified_chroma,
            "image": unified_chroma,
            "path": local_chroma_path,
        }
