"""
Minimal TinyBERT Dual Classifier Inference
Only requires: onnxruntime (pip install onnxruntime)
All other operations done with standard library
"""

import json
import numpy as np
import onnxruntime as ort
from typing import List, Tuple, Dict


class SimpleTokenizer:
    """Lightweight tokenizer without transformers library"""
    
    def __init__(self, vocab_path: str, config_path: str):
        # Load vocabulary
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.max_length = self.config['max_length']
        self.cls_token_id = self.config['cls_token_id']
        self.sep_token_id = self.config['sep_token_id']
        self.pad_token_id = self.config['pad_token_id']
        self.unk_token_id = self.config['unk_token_id']
    
    def tokenize(self, text: str) -> List[str]:
        """Basic WordPiece tokenization"""
        text = text.lower().strip()
        tokens = []
        
        for word in text.split():
            # Check if word is in vocab
            if word in self.vocab:
                tokens.append(word)
            else:
                # Try WordPiece subword tokenization
                start = 0
                sub_tokens = []
                while start < len(word):
                    end = len(word)
                    found = False
                    while start < end:
                        substr = word[start:end]
                        if start > 0:
                            substr = "##" + substr
                        if substr in self.vocab:
                            sub_tokens.append(substr)
                            found = True
                            break
                        end -= 1
                    
                    if not found:
                        sub_tokens.append(self.config['unk_token'])
                        start += 1
                    else:
                        start = end
                
                tokens.extend(sub_tokens)
        
        return tokens
    
    def encode(self, text: str) -> Tuple[np.ndarray, np.ndarray]:
        """Encode text to input_ids and attention_mask"""
        tokens = self.tokenize(text)
        
        # Convert tokens to ids
        token_ids = [self.cls_token_id]
        for token in tokens:
            token_ids.append(self.vocab.get(token, self.unk_token_id))
        token_ids.append(self.sep_token_id)
        
        # Truncate if too long
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length-1] + [self.sep_token_id]
        
        # Create attention mask
        attention_mask = [1] * len(token_ids)
        
        # Pad to max_length
        padding_length = self.max_length - len(token_ids)
        token_ids.extend([self.pad_token_id] * padding_length)
        attention_mask.extend([0] * padding_length)
        
        # Convert to numpy arrays
        input_ids = np.array([token_ids], dtype=np.int64)
        attention_mask = np.array([attention_mask], dtype=np.int64)
        
        return input_ids, attention_mask


class TinyBERTDualClassifierONNX:
    """ONNX-based inference for TinyBERT Dual Classifier"""
    
    def __init__(
        self,
        model_path: str,
        vocab_path: str,
        tokenizer_config_path: str,
        labels_path: str
    ):
        # Load ONNX model
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        
        # Load tokenizer
        self.tokenizer = SimpleTokenizer(vocab_path, tokenizer_config_path)
        
        # Load labels
        with open(labels_path, 'r') as f:
            labels_config = json.load(f)
        
        self.module_labels = labels_config['module_labels']
        self.date_labels = labels_config['date_labels']
    
    def predict(self, text: str) -> Dict[str, str]:
        """Run inference on input text"""
        # Tokenize input
        input_ids, attention_mask = self.tokenizer.encode(text)
        
        # Prepare ONNX inputs
        ort_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
        # Run inference
        module_logits, date_logits = self.session.run(None, ort_inputs)
        
        # Get predictions
        module_pred = int(np.argmax(module_logits, axis=1)[0])
        date_pred = int(np.argmax(date_logits, axis=1)[0])
        
        return {
            'module': self.module_labels[module_pred],
            'date': self.date_labels[date_pred],
            'module_confidence': float(np.max(self._softmax(module_logits[0]))),
            'date_confidence': float(np.max(self._softmax(date_logits[0])))
        }
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Compute softmax values"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def batch_predict(self, texts: List[str]) -> List[Dict[str, str]]:
        """Run inference on multiple texts"""
        return [self.predict(text) for text in texts]


# ==================== USAGE EXAMPLE ====================
if __name__ == "__main__":
    # Initialize model
    print("Loading model...")
    classifier = TinyBERTDualClassifierONNX(
        model_path="tinybert_dual_classifier_quantized.onnx",
        vocab_path="vocab.json",
        tokenizer_config_path="tokenizer_config.json",
        labels_path="labels.json"
    )
    print("Model loaded successfully!\n")
    
    # Single prediction
    test_text = "Show my holdings for this month"
    print(f"Input: {test_text}")
    result = classifier.predict(test_text)
    print(f"Module: {result['module']} (confidence: {result['module_confidence']:.3f})")
    print(f"Date: {result['date']} (confidence: {result['date_confidence']:.3f})")
    
    print("\n" + "="*60)
    
    # Batch prediction
    test_texts = [
        "Show my holdings for this month",
        "What are my capital gains this year?",
        "Display portfolio updates weekly",
        "Show scheme wise returns for previous year"
    ]
    
    print("\nBatch Predictions:")
    print("="*60)
    results = classifier.batch_predict(test_texts)
    
    for text, result in zip(test_texts, results):
        print(f"\nInput: {text}")
        print(f"  → Module: {result['module']} ({result['module_confidence']:.2%})")
        print(f"  → Date: {result['date']} ({result['date_confidence']:.2%})")