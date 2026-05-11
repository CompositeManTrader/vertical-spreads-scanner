# Vertical Spreads Scanner

Scanner EOD para **Put Credit Spreads** sobre SPY y QQQ, basado en research empírico
con backtest 2018-2024 + holdout sellado 2024-2026.

> **Estado**: app en producción para uso personal. NO es asesoramiento financiero.

## Estrategia

Configuración derivada del research (ver carpeta `reports/` privada):

| Parámetro | Valor |
|---|---|
| Subyacentes | SPY, QQQ (NO IWM) |
| Estructura | Put Credit Spread |
| Delta short | -0.30 |
| DTE | 45 |
| Ancho | $5 |
| Take profit | 50% del crédito |
| Stop loss | 2× crédito |
| Time stop | 14 DTE remanentes |
| Filtro entrada | precio > SMA200 **AND** VRP ≥ 3 vol points |
| Sizing | max loss ≤ 2% del capital |

**Performance histórica (train + holdout)**:
- Win rate: 96-100%
- Sharpe per-trade: 1.5-3 (CI bootstrap 1.4-5.0)
- Max DD: $300-450 USD por contrato sobre 6.5 años

## Estructura

```
.
├── streamlit_app.py             # UI principal
├── requirements.txt
├── .streamlit/
│   ├── config.toml              # Theme
│   └── secrets.toml.example     # Template (NO commitear el real)
├── src/
│   ├── scanner/
│   │   ├── data_feed.py         # YFinanceFeed + SchwabFeed
│   │   ├── filter_engine.py     # above_sma200 AND vrp_high
│   │   ├── trade_builder.py     # K_short delta-30 con BSM
│   │   ├── signal_logger.py     # SQLite forward-test
│   │   └── main.py              # CLI scanner (alternativo a UI)
│   ├── pricing/
│   │   └── black_scholes.py     # BSM + solve strike for delta
│   └── analysis/
│       └── indicators.py        # SMA, RV, drawdown (anti-look-ahead)
└── config/
    └── settings.py              # Constantes (dividend yields, costos)
```

## Correr localmente

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

App disponible en `http://localhost:8501`.

## Deploy a Streamlit Cloud

1. Forkear este repo o asegurar que está en tu GitHub.
2. Ir a https://share.streamlit.io → **New app**.
3. Seleccionar este repo, branch `main`, archivo `streamlit_app.py`.
4. (Opcional) En **Advanced settings → Secrets**: pegar el contenido de
   `.streamlit/secrets.toml.example` con tus credenciales reales de Schwab.
5. Click **Deploy**. Live en 1-2 minutos.

## Uso de Schwab API (opcional)

Por defecto la app usa **yfinance** (sin auth, gratis).

Para usar Schwab:

1. Registrar app en https://developer.schwab.com (tarda días/semanas).
2. **Primer auth (interactivo, terminal local)**:
   ```python
   from schwab.auth import easy_client
   c = easy_client(api_key="...", app_secret="...",
                   callback_url="https://127.0.0.1:8182",
                   token_path="./schwab_token.json")
   # Browser se abre, autorizás, copiás URL del redirect.
   ```
3. Setear vars:
   - **Local**: `.env` o `.streamlit/secrets.toml` (ver `.example`).
   - **Streamlit Cloud**: Settings → Secrets.
4. En la UI, sidebar: switch "Data provider" a `schwab`.

## Caveats

- En Streamlit Cloud el filesystem es **ephemeral**: el SQLite log se borra
  con cada redeploy. Para forward-test serio, mover a Postgres/Turso.
- IV de yfinance puede ser ruidosa día a día. Schwab da IV oficial por strike.
- Costos asumidos: $2.80/trade Schwab. Otros brokers ajustar.
- NO operar IWM con esta estrategia.
- Estrategia validada en régimen 2018-2026 (post-QE, COVID, alta inflación).
  Si el régimen cambia drásticamente, re-validar.

## Licencia

Uso personal. No constituye asesoramiento financiero.
