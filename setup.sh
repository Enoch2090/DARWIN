mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"gyc990926@gmail.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
pip3 install torch==1.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
heroku buildpacks:add heroku/jvm
heroku buildpacks:add heroku/java
python3 -m spacy download en_core_web_sm