#!/bin/zsh
python3 -m pyserini.search --topics query.tsv \
                            --index documents \
                            --output run.sample.txt \
                            --bm25