"""
Enhanced Vector Utilities with Smart Token Management and Manufacturing Domain Focus
Production-ready embeddings with advanced retrieval strategies
"""

import asyncio
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import hashlib
import json

# Vector and embedding libraries
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, Range
from sentence_transformers import SentenceTransformer, CrossEncoder
import tiktoken
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Advanced text processing
import networkx as nx
from collections import defaultdict
import spacy
from textstat import flesch_reading_ease

# Monitoring and caching
from functools import lru_cache
import structlog
from retry import retry

# Local imports
from utils import logger

# Configure logger
enhanced_logger = structlog.get_logger()

class TextRankKeywordExtractor:
    """TextRank-based keyword extraction using spaCy and NetworkX."""
    
    def __init__(self, nlp_model):
        self.nlp = nlp_model
        # Manufacturing domain stopwords to filter out
        self.domain_stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'if', 'while', 'at', 'by', 'for', 'with', 
            'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 
            'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 
            'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
            'very', 'can', 'will', 'just', 'should', 'now', 'of', 'is', 'are', 'was', 'were',
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'doing'
        }
        
        # Manufacturing domain POS tags to prioritize
        self.priority_pos = {'NOUN', 'PROPN', 'ADJ', 'VERB'}
        
    def extract_keywords(self, text: str, keyphrase_ngram_range=(1, 3), 
                        stop_words='english', top_k=5) -> List[Tuple[str, float]]:
        """Extract keywords using TextRank algorithm."""
        
        # Process text with spaCy
        doc = self.nlp(text)
        
        # Extract candidate phrases
        candidates = self._extract_candidate_phrases(doc, keyphrase_ngram_range)
        
        if not candidates:
            return []
        
        # Build word graph
        graph = self._build_word_graph(candidates)
        
        # Apply PageRank algorithm
        try:
            scores = nx.pagerank(graph, alpha=0.85, max_iter=100)
        except nx.PowerIterationFailedConvergence:
            # Fallback to degree centrality if PageRank fails
            scores = nx.degree_centrality(graph)
        
        # Rank candidates by score
        ranked_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Convert to expected format (phrase, score)
        results = []
        for phrase, score in ranked_candidates[:top_k]:
            # Normalize score to 0-1 range similar to KeyBERT
            normalized_score = min(score * 2.0, 1.0)  # Scale and cap
            results.append((phrase, normalized_score))
        
        return results
    
    def _extract_candidate_phrases(self, doc, ngram_range=(1, 3)) -> List[str]:
        """Extract candidate phrases from spaCy doc."""
        
        candidates = set()
        
        # Extract n-grams
        for n in range(ngram_range[0], ngram_range[1] + 1):
            for i in range(len(doc) - n + 1):
                # Get n consecutive tokens
                tokens = doc[i:i+n]
                
                # Filter candidates
                if self._is_valid_candidate(tokens):
                    phrase = tokens.text.lower().strip()
                    if phrase and len(phrase) > 2:  # Minimum length
                        candidates.add(phrase)
        
        return list(candidates)
    
    def _is_valid_candidate(self, tokens) -> bool:
        """Check if token sequence is a valid candidate phrase."""
        
        # Must contain at least one priority POS tag
        has_priority_pos = any(token.pos_ in self.priority_pos for token in tokens)
        if not has_priority_pos:
            return False
        
        # Filter out phrases with too many stopwords
        words = [token.text.lower() for token in tokens]
        stopword_ratio = sum(1 for word in words if word in self.domain_stopwords) / len(words)
        if stopword_ratio > 0.5:  # More than 50% stopwords
            return False
        
        # Filter out phrases that are entirely stopwords
        if all(word in self.domain_stopwords for word in words):
            return False
        
        # Filter out single character words (except numbers)
        if len(tokens) == 1 and len(tokens[0].text) == 1 and not tokens[0].is_digit:
            return False
        
        return True
    
    def _build_word_graph(self, candidates: List[str]) -> nx.Graph:
        """Build undirected graph where nodes are candidate phrases."""
        
        graph = nx.Graph()
        graph.add_nodes_from(candidates)
        
        # Create co-occurrence matrix
        window_size = 4  # Words within 4 positions are connected
        
        # Split candidates into words for co-occurrence
        candidate_words = {}
        for candidate in candidates:
            words = candidate.split()
            for word in words:
                if word not in candidate_words:
                    candidate_words[word] = []
                candidate_words[word].append(candidate)
        
        # Build edges based on word co-occurrence
        for word, phrases in candidate_words.items():
            if len(phrases) > 1:
                # Connect all phrases containing this word
                for i, phrase1 in enumerate(phrases):
                    for phrase2 in phrases[i+1:]:
                        if graph.has_edge(phrase1, phrase2):
                            graph[phrase1][phrase2]['weight'] += 1
                        else:
                            graph.add_edge(phrase1, phrase2, weight=1)
        
        # Also add edges based on semantic similarity (simple word overlap)
        for i, phrase1 in enumerate(candidates):
            words1 = set(phrase1.split())
            for phrase2 in candidates[i+1:]:
                words2 = set(phrase2.split())
                overlap = len(words1.intersection(words2))
                if overlap > 0:
                    weight = overlap / max(len(words1), len(words2))  # Jaccard-like similarity
                    if graph.has_edge(phrase1, phrase2):
                        graph[phrase1][phrase2]['weight'] += weight
                    else:
                        graph.add_edge(phrase1, phrase2, weight=weight)
        
        return graph

from pydantic_settings import BaseSettings

@dataclass
class EmbeddingConfig(BaseSettings):
    """Configuration for production embedding system."""
    
    # Model configuration
    embedding_model: str = "BAAI/bge-large-en-v1.5"  # Best for manufacturing/technical content
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2" # Fast and effective reranker
    model_max_length: int = 512
    batch_size: int = 32
    
    # Token management
    max_tokens_per_chunk: int = 400
    token_overlap: int = 50
    enable_token_optimization: bool = True
    
    # Qdrant configuration
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    collection_name: str = "manufacturing_rules_enhanced"
    vector_size: int = 1024  # BGE-large embedding size
    qdrant_persist_path: str = "./qdrant_storage"  # Local persistence path
    prefer_grpc: bool = True  # Use gRPC for better performance
    
    # Advanced retrieval
    enable_hybrid_search: bool = True
    enable_reranking: bool = True
    enable_domain_filtering: bool = True
    
    # Manufacturing domain specifics
    manufacturing_weight: float = 1.5  # Boost manufacturing content
    technical_weight: float = 1.2    # Boost technical content
    context_window_size: int = 3     # Context window for retrieval
    
    # Performance optimization
    enable_caching: bool = True
    cache_size: int = 1000
    batch_processing: bool = True

    class Config:
        env_file = ".env"
        extra = "ignore"
    
class ManufacturingEmbeddingModel:
    """Enhanced embedding model with manufacturing domain optimization."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        
        # Initialize core embedding model
        self.model = SentenceTransformer(config.embedding_model)
        self.model.max_seq_length = config.model_max_length
        
        # Initialize tokenizer for token counting
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Load spaCy model for NER and technical term extraction
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            enhanced_logger.warning("SpaCy model not found, downloading...")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize domain-specific models
        self.keyword_extractor = TextRankKeywordExtractor(self.nlp)
        
        # Manufacturing domain keywords for boosting
        self.manufacturing_keywords = {
            'processes': ['machining', 'molding', 'injection', 'casting', 'assembly', 'welding', 'stamping'],
            'materials': ['steel', 'aluminum', 'plastic', 'polymer', 'composite', 'alloy', 'titanium'],
            'specifications': ['tolerance', 'dimension', 'thickness', 'diameter', 'radius', 'clearance'],
            'quality': ['specification', 'requirement', 'standard', 'minimum', 'maximum', 'shall', 'must'],
            'measurements': ['mm', 'inch', 'cm', 'meter', 'degree', 'micron', 'surface finish']
        }
        
        enhanced_logger.info("Manufacturing embedding model initialized",
                           model=config.embedding_model,
                           max_length=config.model_max_length)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.tokenizer.encode(text))
    
    def extract_manufacturing_features(self, text: str) -> Dict[str, Any]:
        """Extract manufacturing-specific features for domain boosting."""
        
        text_lower = text.lower()
        features = {
            'manufacturing_density': 0.0,
            'technical_complexity': 0.0,
            'keyword_matches': [],
            'entities': [],
            'readability': 0.0
        }
        
        # Calculate manufacturing keyword density
        total_words = len(text.split())
        mfg_words = 0
        
        for category, keywords in self.manufacturing_keywords.items():
            matches = [kw for kw in keywords if kw in text_lower]
            mfg_words += len(matches)
            if matches:
                features['keyword_matches'].extend([(kw, category) for kw in matches])
        
        features['manufacturing_density'] = mfg_words / total_words if total_words > 0 else 0.0
        
        # Extract technical entities using spaCy
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in ['QUANTITY', 'CARDINAL', 'ORDINAL', 'PRODUCT', 'ORG']:
                features['entities'].append((ent.text, ent.label_))
        
        # Calculate technical complexity (number of technical terms / total words)
        technical_terms = len([token for token in doc if token.pos_ in ['NOUN', 'ADJ'] and len(token.text) > 6])
        features['technical_complexity'] = technical_terms / total_words if total_words > 0 else 0.0
        
        # Readability score
        try:
            features['readability'] = flesch_reading_ease(text)
        except:
            features['readability'] = 0.0
        
        return features
    
    def optimize_text_for_embedding(self, text: str) -> str:
        """Optimize text for better embeddings by enhancing manufacturing content."""
        
        if not self.config.enable_token_optimization:
            return text
        
        # Extract features
        features = self.extract_manufacturing_features(text)
        
        # If high manufacturing density, preserve as-is
        if features['manufacturing_density'] > 0.1:
            return text
        
        # For low manufacturing density, try to extract key phrases
        try:
            keywords = self.keyword_extractor.extract_keywords(
                text, 
                keyphrase_ngram_range=(1, 3),
                stop_words='english',
                top_k=5
            )
            
            # Add important keywords to beginning of text for better embedding
            if keywords:
                key_phrases = [kw[0] for kw in keywords if kw[1] > 0.3]
                if key_phrases:
                    enhanced_text = f"Key concepts: {', '.join(key_phrases)}. {text}"
                    return enhanced_text
        except:
            pass
        
        return text
    
    def smart_chunk_with_tokens(self, text: str) -> List[Dict[str, Any]]:
        """Smart chunking with token awareness and manufacturing context preservation."""
        
        # Split into sentences
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        manufacturing_score = 0.0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # Calculate manufacturing relevance of sentence
            sentence_features = self.extract_manufacturing_features(sentence)
            sentence_mfg_score = sentence_features['manufacturing_density']
            
            # Check if adding sentence would exceed token limit
            if current_tokens + sentence_tokens > self.config.max_tokens_per_chunk and current_chunk:
                
                # Optimize chunk text for better embeddings
                optimized_chunk = self.optimize_text_for_embedding(current_chunk.strip())
                
                chunks.append({
                    'text': optimized_chunk,
                    'original_text': current_chunk.strip(),
                    'token_count': current_tokens,
                    'manufacturing_score': manufacturing_score,
                    'sentence_count': len([s for s in current_chunk.split('.') if s.strip()]),
                    'features': self.extract_manufacturing_features(current_chunk)
                })
                
                # Start new chunk with overlap for context preservation
                if self.config.token_overlap > 0:
                    overlap_text = sentences[-1] if sentences else ""
                    current_chunk = overlap_text + " " + sentence
                    current_tokens = self.count_tokens(current_chunk)
                    manufacturing_score = sentence_mfg_score
                else:
                    current_chunk = sentence
                    current_tokens = sentence_tokens
                    manufacturing_score = sentence_mfg_score
            else:
                current_chunk += " " + sentence
                current_tokens += sentence_tokens
                manufacturing_score = max(manufacturing_score, sentence_mfg_score)
        
        # Add final chunk
        if current_chunk.strip():
            optimized_chunk = self.optimize_text_for_embedding(current_chunk.strip())
            chunks.append({
                'text': optimized_chunk,
                'original_text': current_chunk.strip(),
                'token_count': current_tokens,
                'manufacturing_score': manufacturing_score,
                'sentence_count': len([s for s in current_chunk.split('.') if s.strip()]),
                'features': self.extract_manufacturing_features(current_chunk)
            })
        
        # Sort by manufacturing relevance to prioritize important chunks
        chunks.sort(key=lambda x: x['manufacturing_score'], reverse=True)
        
        enhanced_logger.info("Smart chunking completed",
                           total_chunks=len(chunks),
                           avg_tokens=sum(c['token_count'] for c in chunks) / len(chunks) if chunks else 0,
                           avg_mfg_score=sum(c['manufacturing_score'] for c in chunks) / len(chunks) if chunks else 0)
        
        return chunks
    
    @lru_cache(maxsize=1000)
    def encode_cached(self, text: str) -> np.ndarray:
        """Cached encoding for frequently accessed texts."""
        return self.model.encode(text, convert_to_numpy=True)
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Batch encoding with manufacturing domain boosting."""
        
        if not texts:
            return np.array([])
        
        # Optimize texts for better embeddings
        optimized_texts = []
        for text in texts:
            if self.config.enable_token_optimization:
                optimized_text = self.optimize_text_for_embedding(text)
                optimized_texts.append(optimized_text)
            else:
                optimized_texts.append(text)
        
        # Encode in batches
        if len(optimized_texts) <= self.config.batch_size:
            embeddings = self.model.encode(optimized_texts, convert_to_numpy=True, batch_size=self.config.batch_size)
        else:
            # Process in chunks for large batches
            embeddings_list = []
            for i in range(0, len(optimized_texts), self.config.batch_size):
                batch = optimized_texts[i:i + self.config.batch_size]
                batch_embeddings = self.model.encode(batch, convert_to_numpy=True, batch_size=self.config.batch_size)
                embeddings_list.append(batch_embeddings)
            embeddings = np.vstack(embeddings_list)
        
        # Apply domain-specific boosting
        if self.config.enable_domain_filtering:
            embeddings = self._apply_domain_boosting(embeddings, texts)
        
        return embeddings
    
    def _apply_domain_boosting(self, embeddings: np.ndarray, texts: List[str]) -> np.ndarray:
        """Apply manufacturing domain boosting to embeddings."""
        
        boosted_embeddings = embeddings.copy()
        
        for i, text in enumerate(texts):
            features = self.extract_manufacturing_features(text)
            
            # Boost based on manufacturing density
            mfg_boost = 1.0 + (features['manufacturing_density'] * self.config.manufacturing_weight)
            tech_boost = 1.0 + (features['technical_complexity'] * self.config.technical_weight)
            
            # Apply boosting (element-wise multiplication with boost factor)
            boost_factor = min(mfg_boost * tech_boost, 2.0)  # Cap at 2x boost
            boosted_embeddings[i] = boosted_embeddings[i] * boost_factor
            
            # Normalize to maintain embedding space properties
            norm = np.linalg.norm(boosted_embeddings[i])
            if norm > 0:
                boosted_embeddings[i] = boosted_embeddings[i] / norm
        
        return boosted_embeddings

class Reranker:
    """Cross-encoder based reranker for high-precision retrieval."""
    
    def __init__(self, model_name: str):
        try:
            self.model = CrossEncoder(model_name)
            enhanced_logger.info("Reranker initialized", model=model_name)
        except Exception as e:
            enhanced_logger.warning("Failed to initialize reranker", error=str(e))
            self.model = None

    def rerank(self, query: str, docs: List[str], top_k: int = 5) -> List[Tuple[int, float]]:
        """Rerank documents based on query relevance."""
        if not self.model or not docs:
            return [(i, 1.0) for i in range(min(len(docs), top_k))]
            
        pairs = [[query, doc] for doc in docs]
        scores = self.model.predict(pairs)
        
        # Create (index, score) tuples
        results = list(enumerate(scores))
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]

class EnhancedVectorManager:
    """Production-ready vector manager with advanced retrieval strategies."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        
        # Initialize embedding model
        self.embedding_model = ManufacturingEmbeddingModel(config)
        
        # Initialize Reranker
        if config.enable_reranking:
            self.reranker = Reranker(config.reranker_model)
        else:
            self.reranker = None
        
        # Initialize Qdrant client with explicit configuration
        self.client = QdrantClient(
            host=config.qdrant_host,
            port=config.qdrant_port,
            path=config.qdrant_persist_path,
            prefer_grpc=config.prefer_grpc
        )
        
        # Initialize collection
        self._initialize_collection()
        
        # Initialize hybrid search components
        if config.enable_hybrid_search:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 3)
            )
            self.text_corpus = []  # Store texts for TF-IDF
        
        # Cache for frequently accessed vectors
        self.vector_cache = {}
        
        enhanced_logger.info("Enhanced vector manager initialized",
                           collection=config.collection_name,
                           hybrid_search=config.enable_hybrid_search)
    
    def _initialize_collection(self):
        """Initialize Qdrant collection with optimal settings."""
        
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_exists = any(c.name == self.config.collection_name for c in collections)
            
            if not collection_exists:
                # Create collection with optimized settings
                self.client.create_collection(
                    collection_name=self.config.collection_name,
                    vectors_config=VectorParams(
                        size=self.config.vector_size,
                        distance=Distance.COSINE  # Best for semantic similarity
                    )
                )
                enhanced_logger.info("Created new Qdrant collection", 
                                   collection=self.config.collection_name)
            else:
                enhanced_logger.info("Using existing Qdrant collection",
                                   collection=self.config.collection_name)
        
        except Exception as e:
            enhanced_logger.error("Failed to initialize collection", error=str(e))
            raise
    
    @retry(tries=3, delay=1, backoff=2)
    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]] = None) -> List[str]:
        """Add texts to vector database with retry logic."""
        
        if not texts:
            return []
        
        # Generate IDs
        ids = [hashlib.md5(f"{text}_{datetime.utcnow().isoformat()}".encode()).hexdigest() 
               for text in texts]
        
        # Prepare metadata
        if metadatas is None:
            metadatas = [{}] * len(texts)
        
        # Enhance metadata with manufacturing features
        enhanced_metadatas = []
        for i, (text, metadata) in enumerate(zip(texts, metadatas)):
            features = self.embedding_model.extract_manufacturing_features(text)
            enhanced_metadata = {
                **metadata,
                'text_length': len(text),
                'token_count': self.embedding_model.count_tokens(text),
                'manufacturing_density': features['manufacturing_density'],
                'technical_complexity': features['technical_complexity'],
                'readability': features['readability'],
                'added_at': datetime.utcnow().isoformat(),
                'manufacturing_keywords': [kw[0] for kw in features['keyword_matches']],
                'entities': [ent[0] for ent in features['entities']]
            }
            enhanced_metadatas.append(enhanced_metadata)
        
        # Generate embeddings
        embeddings = self.embedding_model.encode_batch(texts)
        
        # Prepare points for Qdrant
        points = []
        for i, (id_, text, embedding, metadata) in enumerate(zip(ids, texts, embeddings, enhanced_metadatas)):
            point = PointStruct(
                id=id_,
                vector=embedding.tolist(),
                payload={
                    'text': text,
                    **metadata
                }
            )
            points.append(point)
        
        # Insert into Qdrant
        try:
            self.client.upsert(
                collection_name=self.config.collection_name,
                points=points
            )
            
            # Update hybrid search corpus if enabled
            if self.config.enable_hybrid_search:
                self.text_corpus.extend(texts)
                # Refit TF-IDF periodically
                if len(self.text_corpus) % 100 == 0:
                    self.tfidf_vectorizer.fit(self.text_corpus)
            
            enhanced_logger.info("Added texts to vector database",
                               count=len(texts),
                               avg_manufacturing_density=sum(m['manufacturing_density'] for m in enhanced_metadatas) / len(enhanced_metadatas))
            
            return ids
            
        except Exception as e:
            enhanced_logger.error("Failed to add texts", error=str(e))
            raise
    
    def similarity_search(self, query: str, top_k: int = 5, 
                         score_threshold: float = 0.0,
                         filter_conditions: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Enhanced similarity search with hybrid retrieval and reranking."""
        
        # 1. Retrieval (Hybrid or Semantic)
        # Fetch more candidates for reranking
        fetch_k = top_k * 4 if self.config.enable_reranking else top_k
        
        if self.config.enable_hybrid_search and len(self.text_corpus) > 10:
            results = self._hybrid_search(query, fetch_k, score_threshold, filter_conditions)
        else:
            results = self._semantic_search(query, fetch_k, score_threshold, filter_conditions)
            
        # 2. Reranking
        if self.config.enable_reranking and self.reranker and results:
            docs = [r['text'] for r in results]
            reranked_indices = self.reranker.rerank(query, docs, top_k=top_k)
            
            final_results = []
            for idx, score in reranked_indices:
                if idx < len(results):
                    res = results[idx]
                    res['rerank_score'] = float(score)
                    # Update similarity score to reflect reranker confidence
                    res['similarity_score'] = float(score) 
                    final_results.append(res)
            
            enhanced_logger.info("Reranking completed", 
                               original_count=len(results), 
                               final_count=len(final_results))
            return final_results
        
        return results[:top_k]
    
    def _semantic_search(self, query: str, top_k: int, score_threshold: float,
                        filter_conditions: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Pure semantic search using embeddings."""
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode_batch([query])[0]
        
        # Prepare filter
        query_filter = None
        if filter_conditions:
            conditions = []
            for field, value in filter_conditions.items():
                if isinstance(value, (int, float)):
                    conditions.append(FieldCondition(key=field, range=Range(gte=value)))
                else:
                    conditions.append(FieldCondition(key=field, match={"value": value}))
            
            if conditions:
                query_filter = Filter(must=conditions)
        
        # Search
        try:
            search_results = self.client.search(
                collection_name=self.config.collection_name,
                query_vector=query_embedding.tolist(),
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=query_filter,
                with_payload=True
            )
            
            # Process results
            results = []
            for result in search_results:
                results.append({
                    'text': result.payload['text'],
                    'similarity_score': result.score,
                    'metadata': {k: v for k, v in result.payload.items() if k != 'text'},
                    'id': result.id
                })
            
            enhanced_logger.info("Semantic search completed",
                               query_preview=query[:50],
                               results_count=len(results),
                               avg_score=sum(r['similarity_score'] for r in results) / len(results) if results else 0)
            
            return results
            
        except Exception as e:
            enhanced_logger.error("Semantic search failed", error=str(e))
            return []
    
    def _hybrid_search(self, query: str, top_k: int, score_threshold: float,
                      filter_conditions: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Hybrid search using Reciprocal Rank Fusion (RRF)."""
        
        # Get semantic results
        semantic_results = self._semantic_search(query, top_k, score_threshold, filter_conditions)
        
        # Get keyword-based results using TF-IDF
        keyword_results = []
        try:
            # Fit TF-IDF if not already fitted
            if not hasattr(self.tfidf_vectorizer, 'vocabulary_'):
                self.tfidf_vectorizer.fit(self.text_corpus)
            
            # Transform query and corpus
            query_tfidf = self.tfidf_vectorizer.transform([query])
            corpus_tfidf = self.tfidf_vectorizer.transform(self.text_corpus)
            
            # Calculate TF-IDF similarities
            tfidf_similarities = cosine_similarity(query_tfidf, corpus_tfidf).flatten()
            
            # Get top TF-IDF results
            top_tfidf_indices = np.argsort(tfidf_similarities)[::-1][:top_k]
            
            for idx in top_tfidf_indices:
                if idx < len(self.text_corpus):
                    text = self.text_corpus[idx]
                    score = float(tfidf_similarities[idx])
                    if score > 0.1: # Min threshold
                        keyword_results.append({
                            'text': text,
                            'similarity_score': score,
                            'id': f"tfidf_{idx}",
                            'metadata': {'source': 'tfidf_keyword'}
                        })
            
        except Exception as e:
            enhanced_logger.warning("Hybrid search (TF-IDF) failed", error=str(e))
            
        # Apply Reciprocal Rank Fusion (RRF)
        # RRF score = 1 / (k + rank)
        k = 60
        scores = defaultdict(float)
        doc_map = {}
        
        # Process semantic results
        for rank, res in enumerate(semantic_results):
            # Use text as key if ID not stable across systems, but here we have IDs
            doc_key = res['text'] # Use text to dedup between semantic and keyword
            scores[doc_key] += 1 / (k + rank + 1)
            doc_map[doc_key] = res
            
        # Process keyword results
        for rank, res in enumerate(keyword_results):
            doc_key = res['text']
            scores[doc_key] += 1 / (k + rank + 1)
            if doc_key not in doc_map:
                doc_map[doc_key] = res
        
        # Sort by RRF score
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        final_results = []
        for doc_key, score in sorted_docs[:top_k]:
            res = doc_map[doc_key]
            res['rrf_score'] = score
            final_results.append(res)
            
        enhanced_logger.info("Hybrid search (RRF) completed",
                           semantic_count=len(semantic_results),
                           keyword_count=len(keyword_results),
                           final_count=len(final_results))
            
        return final_results
    
    def search_with_context(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search with expanded context window for better results."""
        
        # Get initial results
        initial_results = self.similarity_search(query, top_k * 2)
        
        if not initial_results:
            return []
        
        # Expand context for top results
        expanded_results = []
        for result in initial_results[:top_k]:
            
            # Get neighboring chunks for context
            metadata = result['metadata']
            source_file = metadata.get('source_file', '')
            
            if source_file:
                # Search for chunks from same document
                context_results = self.similarity_search(
                    query,
                    top_k=20,
                    filter_conditions={'source_file': source_file}
                )
                
                # Find context around current result
                context_texts = [result['text']]
                for ctx_result in context_results:
                    if (ctx_result['id'] != result['id'] and 
                        ctx_result['similarity_score'] > 0.5):
                        context_texts.append(ctx_result['text'])
                        if len(context_texts) >= self.config.context_window_size:
                            break
                
                # Combine context
                expanded_text = " ... ".join(context_texts)
                
                expanded_results.append({
                    **result,
                    'expanded_text': expanded_text,
                    'context_size': len(context_texts) - 1
                })
            else:
                expanded_results.append(result)
        
        enhanced_logger.info("Context search completed",
                           results_count=len(expanded_results),
                           avg_context_size=sum(r.get('context_size', 0) for r in expanded_results) / len(expanded_results))
        
        return expanded_results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get comprehensive collection statistics."""
        
        try:
            collection_info = self.client.get_collection(self.config.collection_name)
            
            # Get sample of points for analysis
            sample_points = self.client.scroll(
                collection_name=self.config.collection_name,
                limit=100,
                with_payload=True
            )[0]
            
            # Calculate statistics
            stats = {
                'total_points': collection_info.points_count,
                'vector_size': collection_info.config.params.vectors.size,
                'distance_metric': collection_info.config.params.vectors.distance.name,
                'sample_size': len(sample_points)
            }
            
            if sample_points:
                # Analyze metadata
                manufacturing_densities = [p.payload.get('manufacturing_density', 0) for p in sample_points]
                technical_complexities = [p.payload.get('technical_complexity', 0) for p in sample_points]
                text_lengths = [p.payload.get('text_length', 0) for p in sample_points]
                
                stats.update({
                    'avg_manufacturing_density': sum(manufacturing_densities) / len(manufacturing_densities),
                    'avg_technical_complexity': sum(technical_complexities) / len(technical_complexities),
                    'avg_text_length': sum(text_lengths) / len(text_lengths),
                    'high_manufacturing_content': len([d for d in manufacturing_densities if d > 0.1]) / len(manufacturing_densities)
                })
            
            return stats
            
        except Exception as e:
            enhanced_logger.error("Failed to get collection stats", error=str(e))
            return {'error': str(e)}
    
    def optimize_collection(self):
        """Optimize collection performance."""
        
        try:
            # Create index for frequently filtered fields
            self.client.create_payload_index(
                collection_name=self.config.collection_name,
                field_name="manufacturing_density",
                field_schema="float"
            )
            
            self.client.create_payload_index(
                collection_name=self.config.collection_name,
                field_name="source_file",
                field_schema="keyword"
            )
            
            enhanced_logger.info("Collection optimization completed")
            
        except Exception as e:
            enhanced_logger.warning("Collection optimization failed", error=str(e))
    
    def delete_by_filter(self, filter_conditions: Dict[str, Any]) -> int:
        """Delete points by filter conditions."""
        
        try:
            conditions = []
            for field, value in filter_conditions.items():
                conditions.append(FieldCondition(key=field, match={"value": value}))
            
            delete_filter = Filter(must=conditions)
            
            result = self.client.delete(
                collection_name=self.config.collection_name,
                points_selector=delete_filter
            )
            
            enhanced_logger.info("Deleted points", filter=filter_conditions)
            return result.operation_id
            
        except Exception as e:
            enhanced_logger.error("Failed to delete points", error=str(e))
            return 0