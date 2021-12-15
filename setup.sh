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

heroku buildpacks:add heroku/jvm
python3 -m spacy download en_core_web_sm