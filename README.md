# D.A.R.W.I.N.

> DARWIN stands for Documented Automobile Retrieval system With Information Neural network.

## What is Darwin?

Darwin is a natural language car search engine. It allows queries like "I want a car with..." 

Visit [darwin-search.herokuapp.com](https://darwin-search.herokuapp.com) for a live demo.

## Usage

Make sure Java 11 is installed with JVM. Note that Darwin is fully compatible with macOS and Ubuntu, but hasn't been tested on Windows.

```shell
git clone https://github.com/Enoch2090/DARWIN.git darwin
cd darwin
pip install requirements.txt
```

After that, you need to install PyTorch according to your system. `torch-cpu` will also work since the RNN is not large. 

When all dependencies are installed, go to `./interface` and start the server.

```shell
cd ./interface
streamlit run interface.py
```

You can change configurations in `./interface/config.py`. After that, restart the interface to apply the changes.

## Deployment

Darwin is deployable to [Heroku](https://dashboard.heroku.com/apps). However, a few more steps are needed. First, clone this repo to your local machine:

```shell
git clone https://github.com/Enoch2090/DARWIN.git darwin
cd darwin
```

Make sure heroku cli is installed. Run

```shell
heroku login
heroku create NAME_OF_DEPLOYMENT
heroku buildpacks:add heroku/python
heroku buildpacks:add heroku/java
heroku buildpacks:add heroku/jvm
heroku config:add JAVA_HOME=/usr/lib/jvm/java-11-openjdk
git commit -m "deploy"
git push heroku main
```

After the deployment is done, you should see something like this:

```shell
...
remote: -----> Compressing...
remote:        Done: 420.1M
remote: -----> Launching...
remote:  !     Warning: Your slug size (420 MB) exceeds our soft limit (300 MB) which may affect boot time.
remote:        Released v9
remote:        https://darwin-search.herokuapp.com/ deployed to Heroku
remote:
remote: Verifying deploy... done.
To https://git.heroku.com/darwin-search.git
   3237146..7258337  main -> main
Branch 'main' set up to track remote branch 'main' from 'heroku'.
```

Note that you must use `torch-cpu` on Heroku, because the full version has 800MB+ size, which will cause Darwin to exceed the 500MB limit of slug size. The install of `torch-cpu` takes place when Heroku loads the `Procfile`, `setup.sh` will be runned and it will also download the vocab for spaCy.

## About

- [Gu Yucheng](https://github.com/Enoch2090)
- [Duanmu Mingliang](https://github.com/Dmml0621)

Read the [full report](https://github.com/Enoch2090/DARWIN/tree/main/docs/DARWIN.pdf) for detailed info, or check our [blogpost](https://www.enoch2090.me/darwin-a-natural-language-car-information-search-engine). 
