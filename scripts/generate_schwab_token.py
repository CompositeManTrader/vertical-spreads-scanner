"""
Helper para generar el token de Schwab LOCALMENTE y mostrarlo en formato listo
para pegar en Streamlit Cloud Secrets.

Uso:
    python scripts/generate_schwab_token.py

Pre-requisitos:
- Tener .env (raiz del proyecto) con:
    SCHWAB_APP_KEY=...
    SCHWAB_APP_SECRET=...
- O setear esas vars en el shell.

Que hace:
1. Corre easy_client (te abre browser para autorizar Schwab).
2. Genera schwab_token.json en raiz del proyecto.
3. Imprime el contenido en formato TOML, listo para pegar en
   Streamlit Cloud → Settings → Secrets como SCHWAB_TOKEN_JSON.
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv()

APP_KEY = os.getenv("SCHWAB_APP_KEY")
APP_SECRET = os.getenv("SCHWAB_APP_SECRET")
CALLBACK_URL = os.getenv("SCHWAB_CALLBACK_URL", "https://127.0.0.1:8182")
TOKEN_PATH = ROOT / "schwab_token.json"

if not (APP_KEY and APP_SECRET):
    print("ERROR: faltan SCHWAB_APP_KEY o SCHWAB_APP_SECRET en .env")
    sys.exit(1)

print("=" * 70)
print("GENERAR TOKEN DE SCHWAB")
print("=" * 70)
print(f"App key:        {APP_KEY[:8]}...{APP_KEY[-4:]}")
print(f"Callback URL:   {CALLBACK_URL}")
print(f"Token path:     {TOKEN_PATH}")
print()
print("Va a abrir browser. Loguearse en Schwab y autorizar.")
print("Cuando Schwab redirija a 127.0.0.1:8182 (warning de SSL OK), aceptar.")
print("=" * 70)

from schwab.auth import easy_client

client = easy_client(
    api_key=APP_KEY,
    app_secret=APP_SECRET,
    callback_url=CALLBACK_URL,
    token_path=str(TOKEN_PATH),
)

if not TOKEN_PATH.exists():
    print("ERROR: token no se genero.")
    sys.exit(1)

print()
print("=" * 70)
print("TOKEN GENERADO EXITOSAMENTE")
print("=" * 70)
print(f"Archivo: {TOKEN_PATH}")
print()
print("Copiar TODO el siguiente bloque y pegarlo en")
print("Streamlit Cloud → Settings → Secrets:")
print()
print("-" * 70)

token_json = TOKEN_PATH.read_text().strip()
# Re-serializar compacto y validado
parsed = json.loads(token_json)
token_compact = json.dumps(parsed, separators=(",", ":"))

# Usar TOML LITERAL string ''' (single quotes triple) para evitar que TOML
# procese secuencias de escape como \n, \t, \uXXXX dentro del JSON.
print(f'SCHWAB_APP_KEY = "{APP_KEY}"')
print(f'SCHWAB_APP_SECRET = "{APP_SECRET}"')
print('SCHWAB_CALLBACK_URL = "https://127.0.0.1:8182"')
print('SCHWAB_TOKEN_PATH = "/tmp/schwab_token.json"')
print(f"SCHWAB_TOKEN_JSON = '{token_compact}'")
print("-" * 70)
print()
print("CAVEATS:")
print(" - El refresh_token expira a los 7 dias. Re-correr este script semanalmente")
print("   y actualizar el secret en Streamlit Cloud.")
print(" - El access_token expira cada 30 min, schwab-py lo refresca automaticamente.")
print(" - Despues de actualizar secrets en Cloud, click 'Reboot app' para tomar efecto.")
