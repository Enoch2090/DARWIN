from pandas.core.algorithms import rank
from pyserini import index
import streamlit as st
import pandas as pd
import os
import sys
import time
import json
import logging
import base64
import pickle
import shutil
import random
import hashlib
from config import *
from cache import PipelineCache
from torch.autograd import Variable
from pyserini.index import IndexReader
sys.path.insert(1, '..')
from model.rankers import *
from model.feature_predictor import *
from model.feature_extractor import *

# ----------------Logger----------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger.setLevel(level=logging.WARN)
handler = logging.FileHandler('darwin.log')
logger = logging.getLogger("darwin")

# ----------------Functionalities----------------
@st.experimental_memo
def get_query_files():
    return [os.path.join(INTERFACE_CONFIG.query_dir, x) for x in os.listdir(f"{INTERFACE_CONFIG.query_dir}") if ".tsv" in x]

@st.experimental_memo
def get_orig_doc():
    with open(INTERFACE_CONFIG.doc, "r") as f:
        info_json = json.load(f)
    return pd.DataFrame.from_dict(info_json)

@st.experimental_memo
def get_orig_doc_json():
    with open(INTERFACE_CONFIG.doc, "r") as f:
        info_json = json.load(f)
    return info_json

def render_html(html):
    st.markdown(f'{html}', unsafe_allow_html=True)

get_result_key = lambda x: x[1]

@st.experimental_singleton
class Annotation:
    def __init__(self, batch_name, last_object=None):
        self.batch_name = batch_name
        self.doc = get_orig_doc_json()
        with open(f"result_{self.batch_name}.json", "r") as f:
            self.results = json.load(f)
        self.q = 0
        self.r = 0
        self.relevance = []
        logger.info(f'OBJECT Annotation(batch_name="{batch_name}", last_object={last_object}) - created')
        if last_object is not None:
            del last_object
    
    def check_integrity(self, batch_name):
        return batch_name==self.batch_name
    
    def get_next(self, batch_name):
        if self.q==INTERFACE_CONFIG.max_queries:
            st.markdown(f"## {batch_name} finished")
        else:
            logger.info(f'EXEC Annotation.get_next - query {self.q}, result {self.r}')
            if not self.check_integrity(batch_name):
                logger.info(f'EXEC Annotation.get_next - Target changed')
                return Annotation(batch_name=batch_name, last_object=self)
            query = self.results[self.q]["query"]
            result = self.results[self.q]["results"][self.r]
            result_table = self.doc[result["doc_id"]]
            st.markdown(f"Query {self.q+1}/{INTERFACE_CONFIG.max_queries}, Result {self.r+1}/{INTERFACE_CONFIG.max_results}")
            _, button_save, _ = st.columns(3)
            st.markdown(f'## "{query}"')
            st.json(result_table)
            st.markdown(f"Score: {result['score']}")
            st.markdown(f"Is the result relevant?")
            
            button_yes, button_no = st.columns(2)
            is_relevant = False
            
            if button_yes.button(label="Yes"):
                self.update_relevance(True)
            if button_no.button(label="No"):
                self.update_relevance(False)
            
            button_save.button(
                label="Save",
                on_click=self.save()
            )
            
            return self

    def update_relevance(self, value):
        self.relevance.append(
                {
                    "query_id": self.q,
                    "doc_id": self.r,
                    "is_relevant": value
                }
            )
        logger.info(f'EXEC Annotation.update_relevance - query {self.q}, result {self.r} marked as {"relevant" if value else "irrelevant"}')
        self.r += 1
        if self.r == INTERFACE_CONFIG.max_results:
            self.r = 0
            self.q += 1
        if self.q == INTERFACE_CONFIG.max_queries:
            self.save()
        
            
    def save(self):
        with open(f"annotation_{self.batch_name}.json", "w") as f:
            json.dump(self.relevance, f, indent=2)

@st.experimental_singleton
class FeatureExtractorWrapper(FeatureExtractor):
    def __init__(self):
        FeatureExtractor.__init__(self)
    
@st.experimental_singleton
class FeaturePredictorWrapper:
    def __init__(self):
        self.rnn = rnn
        
    def evaluate(self, line_tensor):
        hidden = self.rnn.initHidden()
        for i in range(line_tensor.size()[0]):
            output, hidden = self.rnn(line_tensor[i], hidden)
        return output

    def predict(self, line, n_predictions=3):
        output = self.evaluate(Variable(lineToTensor(line)))
        # Get top N categories
        topv, topi = output.data.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i]
            category_index = topi[0][i]
            # print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value.data.tolist(), all_categories[category_index]])
        logger.info(f"FeaturePredictor <RNN> - Predicted {line} as class {predictions[0][1]} with value {predictions[0][0]}")
        return predictions

@st.experimental_singleton
class RankerWrapper(DemoRanker):
    def __init__(self):
        index_fname = PIPELINE_CONFIG.index_fname
        doc_fname = PIPELINE_CONFIG.doc_fname
        index_reader = IndexReader(index_fname)  # Reading the indexes
        docids, vectors, words_df = get_ids_dict_dfs(doc_fname, index_reader)
        DemoRanker.__init__(self, index_reader, vectors, words_df=words_df, index_fname=index_fname)
        self.length = self.index_reader.stats()["documents"]
    
    def rank(self, query, dict_feature_extraction=None, enhanced=False, big_mtp=0.4, small_mtp=0.1, k1=1.2, b=0.75, k3=1.2):
        scores = []
        query_progress = st.progress(0)
        for i in range(self.length):
            scores.append(
                self.score(
                    query=query,
                    doc_id=str(i),
                    dict_feature_extraction=dict_feature_extraction,
                    enhanced=enhanced,
                    big_mtp=big_mtp,
                    small_mtp=small_mtp,
                    k1=k1,
                    b=b,
                    k3=k3
                )
            )
            query_progress.progress(i/self.length)
        result = list(zip(scores, list(range(self.length))))
        result.sort(reverse=True, key=lambda x: x[0])
        return result
        
@st.experimental_singleton
class DocumentDisplayer:
    def __init__(self):
        self.doc_df = pd.read_json(PIPELINE_CONFIG.doc_fname, lines=True)
        self.doc_df["id"] = self.doc_df["id"].astype('category')
    
    def formatted_print(self, info_dict):
        formatted_output = ""
        for k, v in info_dict.items():
            formatted_output += f"- **{k.replace('_', ' ')}**: {list(v.values())[0]}  \n"
        formatted_output += f"- link: [Go to dealer's page](#)"
        return formatted_output
        
    def show(self, ranking):
        for i, (score, ranked_id) in enumerate(ranking):
            st.markdown(f"### Result {i+1}, score: {score:.4f}")
            st.markdown(f'{self.formatted_print(self.doc_df[self.doc_df["id"]==ranked_id].to_dict())}')
            if i==INTERFACE_CONFIG.max_displayed_results:
                break
            
def cleanse_query(query):
    return query.replace(",", "").replace(".", "").replace("!", "").replace("?", "").split(" ")

# ----------------Options----------------
def option_query(dev_mode):
    st.title("Query")
    query = st.text_input('Enter query', 'I want a car with strong horsepower.')
    enhanced = st.checkbox("Use Feature Extraction", value=True)
    if st.button("Search"):
        with st.spinner("Querying..."):
            query_list = cleanse_query(query)
            if enhanced:
                extracted_features = feature_extractor.extract(query)
                features_dict = {}
                for feature in extracted_features:
                    feature_class = feature_predictor_cache.get(query=feature)
                    if feature_class is None:
                        feature_class = feature_predictor.predict(query, n_predictions=1)[0][1]
                        feature_predictor_cache.update(query=feature, value=feature_class)
                    features_dict[feature] = feature_class
                if dev_mode:
                    st.json(features_dict)
            if enhanced and len(features_dict)>0:
                result = ranker_cache.get(f"{query}-{enhanced}")
                if result is None:
                    result = ranker.rank(query=query_list, dict_feature_extraction=features_dict, enhanced=True)
                ranker_cache.update(f"{query}-{enhanced}", result)
            else:
                result = ranker_cache.get(f"{query}-{enhanced}")
                if result is None:
                    result = ranker.rank(query=query_list, enhanced=False)
                ranker_cache.update(f"{query}-{enhanced}", result)
            document_displayer.show(result)
    return

def option_batch_query(user):
    '''
    Only work as dev_mode
    Selects a query tsv file. Query through the file, return the results, and activate annotation interface.
    '''
    st.title("Batch Query")
    batch_id = f'{user}-{str(time.time()).replace(".", "-")}'
    with st.container():
        uploaded_file = st.file_uploader("Choose a file")
        query_file = None
        if uploaded_file is not None:
            bytes_data = uploaded_file.getvalue()
            query_file = f'query_{batch_id}.tsv'
            with open(query_file, "wb") as f:
                f.write(bytes_data)
        local_file_selection = st.selectbox(
            label="Or select a local file",
            options=get_query_files(),
            index=0
        )
        query_file = local_file_selection if uploaded_file is None else query_file
        st.text(f"Query file is {'uploaded file' if uploaded_file is not None else 'local file'} {query_file}")
        enhanced = st.checkbox("Use Feature Extraction", value=True)
        if st.button("Start"):
            logger.info(f'FUNC option_batch_query(user="{user}") - query started')
            queries = pd.read_csv(query_file, sep="\t", header=None)
            results = []
            for f in os.listdir():
                if "query_" in f and ".csv" in f:
                    os.remove(f)
            queries = queries[1].to_dict().values()
            for j, q in enumerate(queries):
                query_list = cleanse_query(q)
                if enhanced:
                    extracted_features = feature_extractor.extract(q)
                    features_dict = {}
                    for feature in extracted_features:
                        feature_class = feature_predictor_cache.get(query=feature)
                        if feature_class is None:
                            feature_class = feature_predictor.predict(q, n_predictions=1)[0][1]
                            feature_predictor_cache.update(query=feature, value=feature_class)
                        features_dict[feature] = feature_class
                    # st.json(features_dict)
                if enhanced and len(features_dict)>0:
                    result = ranker.rank(query=query_list, dict_feature_extraction=features_dict, enhanced=True) # Batch querying does not use cache
                else:
                    result = ranker.rank(query=query_list, enhanced=False)
                results.append(
                    {
                        "query": q,
                        "results": [
                            {
                                "doc_id": x[1],
                                "score": x[0]
                            } for x in result[0:INTERFACE_CONFIG.max_results]
                        ]
                    }
                )
            with open(f'result_{batch_id}.json', "w") as f:
                json.dump(results, f, indent=2)
            st.success(f"Batch {batch_id} success")
            logger.info(f'FUNC option_batch_query(user="{user}") - query success')        
    return

def option_annotate(batch_name):
    st.title("Annotate")
    annotate = Annotation(batch_name=batch_name)
    annotate = annotate.get_next(batch_name=batch_name)            

def option_edit_cache():
    st.title("Edit Cache")
    if INTERFACE_CONFIG.cache_edit:
        st.markdown("**Important warning:** Cache editing function calls `eval()`, do not expose this function to the web!") 
        from streamlit_ace import st_ace
        cache_file = st.selectbox(
            "Cache File",
            [os.path.join(CACHE_CONFIG.cache_dir, x) for x in os.listdir(f"{CACHE_CONFIG.cache_dir}") if ".pkl" in x] 
        )
        with open(cache_file, "rb") as f:
            cache_json = json.dumps(pickle.load(f), indent=2)
            logger.info(f'FUNC option_edit_cache() - {cache_file} loaded')
        cache_json_updated = st_ace(value=cache_json, language="python", theme="github")
        if cache_json_updated:
            with open(cache_file, "wb") as f:
                pickle.dump(eval(cache_json_updated), f)
                logger.info(f'FUNC option_edit_cache() - {cache_file} saved')
    else:
        st.markdown("Cache edit not enabled in configuration!")       
            
          
def option_about():
    with open('../README.md', 'r') as f:
        readme = "".join(f.readlines())
    st.markdown(readme)

# ----------------Menu----------------
feature_extractor = FeatureExtractorWrapper()
feature_extractor_cache = None # no need to add a cache for this version
feature_predictor = FeaturePredictorWrapper()
feature_predictor_cache = PipelineCache(name="FeaturePredictorCache")
ranker = RankerWrapper()
ranker_cache = PipelineCache(name="RankerCache")
document_displayer = DocumentDisplayer()
st.sidebar.title('D.A.R.W.I.N.')
st.sidebar.markdown(f"**D**ocumented **A**utomobile **R**etrieval system **W**ith **I**nformation **N**eural network")
if INTERFACE_CONFIG.dev_mode:
    dev_mode = st.sidebar.checkbox("Development Mode", value=False)
else:
    dev_mode = False
option = st.sidebar.selectbox(
            'Menu',
            ['Query', 'About']
        ) \
        if not dev_mode else \
        st.sidebar.selectbox(
            'Menu',
            ['Query', 'Batch Query', 'Annotate', 'Edit Feature Cache']
        )
if dev_mode:
    user = st.sidebar.selectbox(
            'User',
            INTERFACE_CONFIG.users
        )
    # st.sidebar.markdown(f"Other users running: **{False}**")
if option == 'Query':
    option_query(dev_mode=dev_mode)
elif option == 'Batch Query':
    option_batch_query(user)
elif option == 'Annotate':
    batch_name = st.sidebar.selectbox(
        label="Select cached results",
        options = [x.replace("result_", "").replace(".json", "") for x in os.listdir() if user in x and "result" in x and "json" in x]
    )
    logger.info(f'EXEC option_annotate(batch_name="{batch_name}")')
    option_annotate(batch_name=batch_name)
elif option == 'Edit Feature Cache':
    option_edit_cache()
elif option == 'About':
    option_about()


# ----------------Hide Development Menu----------------
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)
