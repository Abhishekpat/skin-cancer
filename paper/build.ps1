# =====================================================================
#  build.ps1  --  Build the Skin Cancer Detection IEEE paper (Windows).
#     ./build.ps1        # both PDFs
#     ./build.ps1 two    # two-column only
#     ./build.ps1 one    # one-column only
#  Engine preference: tectonic -> latexmk -> pdflatex (+bibtex).
#  Or upload *.tex + references.bib to Overleaf and compile
#  sc_two_column.tex as the main file.
# =====================================================================
param([string]$target = "all")
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$papers = @{ "one" = "sc_one_column"; "two" = "sc_two_column" }

function Get-Engine {
    foreach ($e in "tectonic","latexmk","pdflatex") {
        if (Get-Command $e -ErrorAction SilentlyContinue) { return $e }
    }
    return $null
}

function Build-One([string]$name) {
    $engine = Get-Engine
    if (-not $engine) {
        Write-Host "No LaTeX engine found. Install Tectonic (https://tectonic-typesetting.github.io)," -ForegroundColor Red
        Write-Host "MiKTeX, or TeX Live -- or compile on https://overleaf.com."
        exit 1
    }
    Write-Host ">> Building $name.pdf with $engine" -ForegroundColor Cyan
    switch ($engine) {
        "tectonic" { & tectonic --chatter minimal "$name.tex" }
        "latexmk"  { & latexmk -pdf -interaction=nonstopmode "$name.tex" }
        "pdflatex" {
            & pdflatex -interaction=nonstopmode "$name.tex"
            & bibtex   "$name"
            & pdflatex -interaction=nonstopmode "$name.tex"
            & pdflatex -interaction=nonstopmode "$name.tex"
        }
    }
}

switch ($target) {
    "one"   { Build-One $papers["one"] }
    "two"   { Build-One $papers["two"] }
    default { Build-One $papers["two"]; Build-One $papers["one"] }
}
