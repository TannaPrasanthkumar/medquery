import logging
from sentence_transformers import SentenceTransformer
from config.config import EMBEDDING_MODEL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicalEmbeddingModel:
    
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def load(self):
        """Load the sentence-transformer model (called once)."""
        if not self._loaded:
            try:
                logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
                self.model = SentenceTransformer(EMBEDDING_MODEL)
                self._loaded = True
                logger.info("Embedding model loaded successfully.")
            except Exception as e:
                logger.error("Failed to load embedding model: %s", e)
                raise

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of text strings.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors (list of floats).
        """
        try:
            if not self._loaded:
                self.load()
            embeddings = self.model.encode(texts, convert_to_list=True, show_progress_bar=False)
            return embeddings
        except Exception as e:
            logger.error("embed_texts error: %s", e)
            raise

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query string.

        Args:
            query: The search query string.

        Returns:
            A single embedding vector.
        """
        try:
            if not self._loaded:
                self.load()
            embedding = self.model.encode([query], convert_to_list=True, show_progress_bar=False)
            return embedding[0]
        except Exception as e:
            logger.error("embed_query error: %s", e)
            raise


def get_embedding_model() -> MedicalEmbeddingModel:
    """
    Returns a loaded embedding model instance.
    Uses singleton pattern — model loads only once per session.
    """
    try:
        model = MedicalEmbeddingModel()
        model.load()
        return model
    except Exception as e:
        logger.error("get_embedding_model failed: %s", e)
        raise
