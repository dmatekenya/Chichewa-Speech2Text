
# build jupyter book
jupyter-book build . --config docs/_config.yml --toc docs/_toc.yml

# open to review the book
open _build/html/index.html

# Copy it to docs folder
cp -r _build/*  docs/_build/

# publish 
ghp-import -n -p -f docs/_build/html
