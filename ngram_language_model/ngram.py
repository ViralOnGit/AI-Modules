import re
from collections import defaultdict, Counter
import sys

class NgramCharacterModel:
    def __init__(self, corpus, n):
        # Initialize the class variables
        self.n = n
        self.ngram_counts = defaultdict(int)
        self.context_counts = defaultdict(int)
        self.vocabulary = set()
        self.words = set()
        self._train(corpus)
    
    def _train(self, corpus):
        corpus = re.sub(r'[^\w\s]', ' ', corpus.lower())
        words = re.findall(r'\b\w+\b', corpus)
        self.words = set(words)
        marked_words = []
        for word in words:
            # Add start and end markers
            marked_word = '^' * (self.n - 1) + word + '$'
            marked_words.append(marked_word)
            
            # Add characters to vocabulary
            for char in word:
                self.vocabulary.add(char)
        
        # Count n-grams and their contexts
        for word in marked_words:
            for i in range(len(word) - self.n + 1):
                ngram = word[i:i+self.n]
                self.ngram_counts[ngram] += 1
                context = word[i:i+self.n-1]
                self.context_counts[context] += 1

    def _generate_word(self, prefix):
        # Given a prefix, generate the most probable word that it completes to
        # If prefix is empty, return empty string
        if not prefix:
            return ""
        
        # Start with the prefix
        word = prefix
        max_length = 50  # Avoid infinite loops
        
        # Continue generating characters until we reach an end marker or max length
        context = word[-(self.n-1):] if len(word) >= self.n-1 else '^' * (self.n-1 - len(word)) + word
        
        while len(word) < max_length:
            next_char_probs = {}
            for ngram in self.ngram_counts:
                if ngram.startswith(context):
                    next_char = ngram[-1]
                    prob = self.ngram_counts[ngram] / self.context_counts[context]
                    next_char_probs[next_char] = prob
            
            if not next_char_probs:
                break
                
            # Select the character with highest probability
            next_char = max(next_char_probs, key=next_char_probs.get)
            
            # If we reach end marker, we're done
            if next_char == '$':
                break
                
            word += next_char
            context = word[-(self.n-1):] if len(word) >= self.n-1 else word
        
        return word

    def predict_top_words(self, prefix, top_k=10):
        # Given a prefix, return the top_k most probable words from the corpus it completes to
        if not prefix:
            return []
        
        # Clean the prefix
        prefix = prefix.lower()
        
        # Find matching words from corpus
        matching_words = [word for word in self.words if word.startswith(prefix)]
        
        # If we have exact matches in our corpus
        if matching_words:
            # Calculate probability for each word
            word_probs = [(word, self._word_probability(word)) for word in matching_words]
            # Sort by probability (descending)
            word_probs.sort(key=lambda x: x[1], reverse=True)
            # Return top_k words
            return [word for word, _ in word_probs[:top_k]]
        else:
            # Generate a word if no exact matches
            generated_word = self._generate_word(prefix)
            return [generated_word] if generated_word else []
    
    def _word_probability(self, word):
        # Calculates the probability of the word, based on the n-gram probabilities
        if not word:
            return 0
        
        # Add start and end markers
        marked_word = '^' * (self.n - 1) + word + '$'
        
        # Calculate probability as product of n-gram probabilities
        probability = 1.0
        for i in range(len(marked_word) - self.n + 1):
            ngram = marked_word[i:i+self.n]
            context = marked_word[i:i+self.n-1]
            
            # Avoid division by zero
            if self.context_counts[context] == 0:
                return 0
            
            # Multiply by conditional probability
            probability *= self.ngram_counts[ngram] / self.context_counts[context]
        
        return probability
    
