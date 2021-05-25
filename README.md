# Anime face generation using DCGAN

> An interesting and special GAN method consisting of convolutional layers to generate pixel perfect anime images.

## Installation and usage.

This project uses `pipenv` for dependency management. You need to ensure that you have `pipenv`
installed on your system.

Here's how to install the dependencies, and get started.

- Install it using **`pipenv sync -d`**
- Once done, spawn a shell to run the files: `pipenv shell`

And you're done, and you can run any of the files, and test them.

### Project structure

This project has 3 main sections.

- `src/` Contains the python scripts for training the ML Models.
- `notebooks/` contains the jupyter notebooks with explanations and the outputs of our end
  goal.
- `models/` contains the exported model to make your work easy.

### Generating datasets

I have personally used some kaggle datasets to train the model. You can also use the
script as I have provided in the repository, or make your own.

To make your own, Here are the steps. You need a scraper tool called [gallery-dl](https://github.com/mikf/gallery-dl)
to download the images first, and then use [Animeface](https://github.com/nya3jp/python-animeface)
to process the images and get the faces.

- Download the images using `gallery-dl`. Here is a python script to automate it. Note, You need the tags
  in a file called `tags.txt` in same folder. The tags are pre-added in the repo, inside the `resources`
  folder.
  ```python
  import os

  with open("tags.txt", "r") as f:
    tags = f.read()

  for tag in tags.split("\n"):
    os.system(f'gallery-dl --images 1000 "https://danbooru.donmai.us/posts?tags={tag}"')
  ```
- Once done, You can follow the anime face documentation and process the data and then build the
  dataset out of it.

## Contributing

Contributions, issues and feature requests are welcome. After cloning & setting up project locally, you
can just submit a PR to this repo and it will be deployed once it's accepted.

⚠️ It’s good to have descriptive commit messages, or PR titles so that other contributors can understand about your
commit or the PR Created. Read [conventional commits](https://www.conventionalcommits.org/en/v1.0.0-beta.3/)
before making the commit message.

## Show your support

We love people's support in growing and improving. Be sure to leave a ⭐️ if you like the project and
also be sure to contribute, if you're interested!

<div align="center">
Made by Sunrit Jana with <3
</div>
