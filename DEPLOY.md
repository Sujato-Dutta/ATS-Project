# Deployment Instructions for Streamlit Cloud

## 1. Prepare the Repository
Run the following commands in your terminal to commit the new demo app and mock data:

```bash
git add requirements.txt src/demo_app.py src/generate_mock_data.py mock_outputs/
git commit -m "Add demo app and mock data for presentation"
git push origin main
```

## 2. Deploy to Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io/).
2. Click **"New app"**.
3. Select your repository (`Time Series Project`).
4. **Main file path**: Enter `src/demo_app.py`.
5. Click **"Deploy"**.

## Notes
- The app uses pre-generated data from `mock_outputs/` so it will load instantly.
- The heavy `data/` folder is ignored, which is correct.
- `requirements.txt` has been updated to include `plotly`.
