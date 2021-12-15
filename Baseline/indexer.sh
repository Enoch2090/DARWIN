#!/bin/zsh
python3 -m pyserini.index -collection JsonCollection \
                         -generator DefaultLuceneDocumentGenerator \
                         -threads 4 \
                         -input ./documents \
                         -index ./documents \
                         -storePositions -storeDocvectors -storeRaw