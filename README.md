```
├── README.md       <- this file
├── data            <- store your data here
│   ├── final         <- results
│   ├── processed     <- preprocessed data
│   ├── raw           <- raw, initial data
│   └── sim           <- simulated data
├── dev             <- used during development
│   ├── notebooks     <- jupyter / pluto notebooks
│   └── scripts       <- demo scripts
├── docs            <- documentation
├── references      <- other research papers
├── reports         <- your report, presentation, etc
│   ├── img           <- image folder for report
│   └── report.qmd    <- empty reportfile. You can use .qmd or .tex
└── src
```

This is a setup for the practical assignment for the Machine Learning course. I created the template with https://github.com/raoulg/DsTemplate

There is a Makefile that installs some things I like for my environment:
`make all` will install [https://starship.rs/](starship) (and some nerdfonts), auto-suggestions and [zoxide](https://github.com/ajeetdsouza/zoxide).
You can skip that if you want to, but I would recommend still to do `make add-path` if you want to skip it.

I also added `make format` and `make lint` to run some linting on the `src` and `dev/scripts` folders. Note that running `make format` possible solves errors that `make lint` will find.

For documentation, you can use
- [documenter.jl](https://documenter.juliadocs.org/stable/) for julia
- [sphinx](https://www.sphinx-doc.org/en/master/) for python

For the report, you can either use
- [Quarto](https://quarto.org/) which is relative easy
- [latex](https://www.latex-project.org/get/) which is much more complex but gives better typesetting for more professional looking reports.
