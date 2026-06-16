python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

CMEMS_USER=claypotdat01@gmail.com
CMEMS_PASS=limaJuni_2026
NASA_USER=claypotdat1
NASA_PASS=limaJuni_2026
CDS_KEY=e6b04d44-1240-4812-b642-a49e762e49b5
BMKG_ADM4=81.71.01.1001

streamlit run app.py
