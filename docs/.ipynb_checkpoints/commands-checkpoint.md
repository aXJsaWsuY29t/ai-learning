## Create Python env (Windows/PowerShell)
```bash
python -m venv .venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.venv\Scripts\activate
python.exe -m pip install --upgrade pip
```


## Install Python packages
```bash
pip install -r requirements.txt
```


## Run tutorials
```bash
cd .\01-linear-algebra-softmax
pytest test/
```


## Open notebook for notes
```bash
jupyter notebook notes.ipynb
```
