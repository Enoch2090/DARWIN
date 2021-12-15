from treelib import Node, Tree
import spacy
from spacy.matcher import DependencyMatcher
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex
import logging
nlp = spacy.load('en_core_web_lg')

def custom_tokenizer(nlp):
    inf = list(nlp.Defaults.infixes)               # Default infixes
    inf.remove(r"(?<=[0-9])[+\-\*^](?=[0-9-])")    # Remove the generic op between numbers or between a number and a -
    inf = tuple(inf)                               # Convert inf to tuple
    infixes = inf + tuple([r"(?<=[0-9])[+*^](?=[0-9-])", r"(?<=[0-9])-(?=-)"])  # Add the removed rule after subtracting (?<=[0-9])-(?=[0-9]) pattern
    infixes = [x for x in infixes if '-|–|—|--|---|——|~' not in x] # Remove - between letters rule
    infix_re = compile_infix_regex(infixes)

    return Tokenizer(nlp.vocab, prefix_search=nlp.tokenizer.prefix_search,
                                suffix_search=nlp.tokenizer.suffix_search,
                                infix_finditer=infix_re.finditer,
                                token_match=nlp.tokenizer.token_match,
                                rules=nlp.Defaults.tokenizer_exceptions)
class FeatureExtractor:
    def __init__(self):
        self.dep_matcher = DependencyMatcher(vocab=nlp.vocab)
        main_patterns = [
                # [
                #     {'RIGHT_ID': 'p_subject', 'RIGHT_ATTRS': {'TEXT': 'car', 'DEP': 'dobj'}},
                #     {'LEFT_ID': 'p_subject', 'REL_OP': '>', 'RIGHT_ID': 'p_prep_to_subject', 'RIGHT_ATTRS': {'DEP': 'prep'}},
                #     {'LEFT_ID': 'p_prep_to_subject', 'REL_OP': '>', 'RIGHT_ID': 'p_prep_object', 'RIGHT_ATTRS': {'DEP': {"REGEX": "\s*"}}}
                # ],
                [
                {'RIGHT_ID': 'p_object', 'RIGHT_ATTRS': {'DEP': {"IN": ['dobj', 'pobj']}}},
                {'LEFT_ID': 'p_object', 'REL_OP': '>', 'RIGHT_ID': 'p_object_mod', 'RIGHT_ATTRS': {'DEP': 'amod'}},
                ]
            ] # TODO: CONFIG HERE
        self.dep_matcher.add(f"main_patterns", patterns=main_patterns)
        self.logger = logging.getLogger("darwin")
        
    @staticmethod
    def traverse(s, tree):
        for c in s.children:
            tree.create_node(c, c,  parent=s, data=c)
            tree = FeatureExtractor.traverse(c, tree=tree)
        return tree
    
    @staticmethod   
    def build_tree(query):
        doc = nlp(query)
        sent_tree = Tree()
        s = list(doc.sents)[0].root
        sent_tree.create_node(s, s, data=s)
        sent_tree = FeatureExtractor.traverse(s, sent_tree)
        return sent_tree
    
    @staticmethod  
    def build_tree_from_node(node):
        sent_tree = Tree()
        sent_tree.create_node(node, node, data=node)
        sent_tree = FeatureExtractor.traverse(node, sent_tree)
        return sent_tree
    
    @staticmethod
    def compile_secondary_patterns(main_subject=""):
        secondary_patterns = [
            [
                {'RIGHT_ID': 'p_object', 'RIGHT_ATTRS': {'DEP': {"IN": ['dobj', 'pobj']}}},
                {'LEFT_ID': 'p_object', 'REL_OP': '>', 'RIGHT_ID': 'p_object_mod', 'RIGHT_ATTRS': {'DEP': {"IN": ['amod', 'compound']}}},
            ],
            [
                {'RIGHT_ID': 'p_conj', 'RIGHT_ATTRS': {'DEP': 'conj'}},
                {'LEFT_ID': 'p_conj', 'REL_OP': '>', 'RIGHT_ID': 'p_conj_mod', 'RIGHT_ATTRS': {'DEP': {"IN": ['amod', 'advmod']}}},
            ]
        ]
        return secondary_patterns
    
    def log(self, log, level="info"):
        if self.logger:
            if level=="info":
                self.logger.info(log)
            elif level=="warning":
                self.logger.warning(log)
            elif level=="error":
                self.logger.error(log)
            elif level=="debug":
                self.logger.debug(log)
            elif level=="critical":
                self.logger.critical(log)
    
    def extract(self, query):
        nlp.tokenizer = custom_tokenizer(nlp)
        doc = nlp(query)
        secondary_matcher = DependencyMatcher(vocab=nlp.vocab)
        secondary_matcher.add("secondary_patterns", FeatureExtractor.compile_secondary_patterns())
        dep_matches = secondary_matcher(doc)
        matches_str = []
        for match in dep_matches:
            matches = match[1]
            p_1, p_2 = matches[0], matches[1]
            # print(f"\t-> {doc[p_1]} {doc[p_2]}")
            matches_str.append(f"{doc[p_2]} {doc[p_1]}")
        self.log(f"CLASS FeatureExtractor - FUNC extract(self, query={query}) - extracts {len(matches_str)} matches")
        return matches_str        