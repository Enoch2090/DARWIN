#!/bin/zsh
python3 -m pyserini.index -collection JsonCollection \
                         -generator DefaultLuceneDocumentGenerator \
                         -threads 4 \
                         -input ./data \
                         -index ./index \
                         -storePositions -storeDocvectors -storeRaw