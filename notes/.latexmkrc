@default_files = ('main.tex');
$bibtex = 'bibtex';
$pdflualatex = 'lualatex -interaction=nonstopmode -synctex=1 %O %S';
$pdf_mode = 4;  # use lualatex to make pdf
$success_cmd = "latexmk -c main.tex";
$aux_dir = "aux";
$ENV{'LUAOTFLOAD_CACHE'} = "$aux_dir/luacache";
$ENV{'TEXMFVAR'} = "$aux_dir/texmf-var";
$ENV{'TEXMFCACHE'} = "$aux_dir/texmf-var";
