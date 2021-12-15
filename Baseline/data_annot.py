import streamlit as st
import pandas as pd
import os
import sys
import time
import json
import base64
import pickle
import shutil
import random
import hashlib

QUERY_SOURCE = "query.tsv"
DOCUMENT_SOURCE = "joined.csv"

# ----------------Menu----------------



# ----------------Hide Development Menu----------------
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)
