"""
Example API client for Medical AI Assistant.
Demonstrates how to interact with the API endpoints.
"""

import requests
import json
from typing import Dict, Any, List


class MedicalAIClient:
    """Client for Medical AI Assistant API."""
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = None, token: str = None):
        """
        Initialize API client.
        
        Args:
            base_url: Base URL of the API
            api_key: API key for authentication
            token: JWT token for authentication
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.token = token
        self.session = requests.Session()
        
        # Set authentication headers
        if api_key:
            self.session.headers['X-API-Key'] = api_key
        elif token:
            self.session.headers['Authorization'] = f'Bearer {token}'
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request to API."""
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error: {e}")
            print(f"Response: {e.response.text}")
            raise
        except Exception as e:
            print(f"Error: {e}")
            raise
    
    # Authentication
    def login(self, username: str, password: str) -> str:
        """
        Login and get JWT token.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            str: JWT access token
        """
        response = self._make_request(
            'POST',
            '/api/v1/auth/token',
            json={'username': username, 'password': password}
        )
        
        self.token = response['access_token']
        self.session.headers['Authorization'] = f"Bearer {self.token}"
        return self.token
    
    # Health check
    def health_check(self) -> Dict[str, Any]:
        """Get API health status."""
        return self._make_request('GET', '/health')
    
    # NER endpoints
    def extract_entities(
        self,
        text: str,
        model_type: str = "transformer",
        include_linking: bool = False
    ) -> Dict[str, Any]:
        """
        Extract medical entities from text.
        
        Args:
            text: Input text
            model_type: Model type ('transformer' or 'bilstm_crf')
            include_linking: Include entity linking
            
        Returns:
            Dict: NER response with entities
        """
        return self._make_request(
            'POST',
            '/api/v1/ner/extract',
            json={
                'text': text,
                'model_type': model_type,
                'include_linking': include_linking
            }
        )
    
    def extract_entities_batch(
        self,
        texts: List[str],
        model_type: str = "transformer"
    ) -> Dict[str, Any]:
        """
        Extract entities from multiple texts.
        
        Args:
            texts: List of texts
            model_type: Model type
            
        Returns:
            Dict: Batch NER response
        """
        return self._make_request(
            'POST',
            '/api/v1/ner/extract/batch',
            json={
                'texts': texts,
                'model_type': model_type
            }
        )
    
    # Classification endpoints
    def classify_text(
        self,
        text: str,
        top_k: int = 3,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Classify clinical text.
        
        Args:
            text: Input text
            top_k: Number of predictions
            threshold: Confidence threshold
            
        Returns:
            Dict: Classification response
        """
        return self._make_request(
            'POST',
            '/api/v1/classification/predict',
            json={
                'text': text,
                'top_k': top_k,
                'threshold': threshold
            }
        )
    
    def get_categories(self) -> Dict[str, Any]:
        """Get available classification categories."""
        return self._make_request('GET', '/api/v1/classification/categories')
    
    # RAG endpoints
    def ask_question(
        self,
        query: str,
        top_k: int = 5,
        use_chain_of_thought: bool = False,
        conversation_id: str = None
    ) -> Dict[str, Any]:
        """
        Ask a medical question.
        
        Args:
            query: Question to ask
            top_k: Number of documents to retrieve
            use_chain_of_thought: Use CoT reasoning
            conversation_id: Conversation ID for context
            
        Returns:
            Dict: RAG response with answer
        """
        return self._make_request(
            'POST',
            '/api/v1/rag/ask',
            json={
                'query': query,
                'top_k': top_k,
                'use_chain_of_thought': use_chain_of_thought,
                'conversation_id': conversation_id
            }
        )
    
    def upload_document(
        self,
        content: str,
        metadata: Dict[str, Any] = None,
        source: str = None
    ) -> Dict[str, Any]:
        """
        Upload document to RAG system.
        
        Args:
            content: Document content
            metadata: Document metadata
            source: Document source
            
        Returns:
            Dict: Upload response
        """
        return self._make_request(
            'POST',
            '/api/v1/rag/documents/upload',
            json={
                'content': content,
                'metadata': metadata or {},
                'source': source
            }
        )
    
    # Safety endpoints
    def check_safety(
        self,
        text: str,
        check_type: str = "both"
    ) -> Dict[str, Any]:
        """
        Check text for safety issues.
        
        Args:
            text: Text to check
            check_type: Check type ('query', 'response', or 'both')
            
        Returns:
            Dict: Safety check response
        """
        return self._make_request(
            'POST',
            '/api/v1/safety/check',
            json={
                'text': text,
                'check_type': check_type
            }
        )
    
    def get_emergency_criteria(self) -> Dict[str, Any]:
        """Get emergency detection criteria."""
        return self._make_request('GET', '/api/v1/safety/emergency/criteria')


def main():
    """Example usage of the API client."""
    
    print("=" * 80)
    print("MEDICAL AI ASSISTANT - API CLIENT EXAMPLES")
    print("=" * 80)
    
    # Initialize client
    # Option 1: Using API key
    client = MedicalAIClient(
        base_url="http://localhost:8000",
        api_key="your-api-key-here"
    )
    
    # Option 2: Using username/password to get token
    # token = client.login("demo", "demo123")
    # print(f"Logged in with token: {token[:20]}...")
    
    print("\n1. Health Check")
    print("-" * 80)
    try:
        health = client.health_check()
        print(f"Status: {health['status']}")
        print(f"Uptime: {health['uptime_seconds']:.2f}s")
        print(f"Services: {health['services']}")
    except Exception as e:
        print(f"Health check failed: {e}")
    
    print("\n2. NER - Entity Extraction")
    print("-" * 80)
    try:
        ner_result = client.extract_entities(
            text="Patient has hypertension and takes lisinopril 10mg daily.",
            model_type="transformer"
        )
        print(f"Found {len(ner_result['entities'])} entities:")
        for entity in ner_result['entities']:
            print(f"  - {entity['text']} ({entity['label']}) [{entity['start']}:{entity['end']}]")
    except Exception as e:
        print(f"NER failed: {e}")
    
    print("\n3. Classification")
    print("-" * 80)
    try:
        class_result = client.classify_text(
            text="Patient presents with chest pain, shortness of breath, and elevated troponin levels.",
            top_k=3
        )
        print("Top predictions:")
        for pred in class_result['predictions']:
            print(f"  - {pred['label']}: {pred['probability']:.3f}")
    except Exception as e:
        print(f"Classification failed: {e}")
    
    print("\n4. RAG - Question Answering")
    print("-" * 80)
    try:
        rag_result = client.ask_question(
            query="What are the first-line treatments for hypertension?",
            top_k=5
        )
        print(f"Answer: {rag_result['answer'][:200]}...")
        print(f"Sources: {rag_result['num_sources']}")
        print(f"Disclaimers: {len(rag_result['disclaimers'])}")
    except Exception as e:
        print(f"RAG failed: {e}")
    
    print("\n5. Safety Check")
    print("-" * 80)
    try:
        safety_result = client.check_safety(
            text="I'm having severe chest pain and can't breathe",
            check_type="query"
        )
        print(f"Safety Level: {safety_result['safety_level']}")
        print(f"Emergency Detected: {safety_result['emergency_detected']}")
        if safety_result['warnings']:
            print("Warnings:")
            for warning in safety_result['warnings']:
                print(f"  - {warning}")
    except Exception as e:
        print(f"Safety check failed: {e}")
    
    print("\n" + "=" * 80)
    print("API CLIENT EXAMPLES COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
