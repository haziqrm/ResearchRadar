# Research Radar

A map of quantitative finance research: which topics are growing, which are fading, and how they connect. Everything runs on free public data.

## What it does

- **Topic map.** Each circle is a research topic, sized by its number of papers. The lines between topics show papers that cite the same earlier work. Drag the slider (or press play) to watch topics grow and shrink, by year (2000 to now) or by month (recent arXiv papers).
- **Inside a topic.** Click a circle to see its sub-topics as clusters of individual papers, the citations between them, and suggested reading.
- **Where research is heading.** A ranking of which topics are likely to grow over the next three years.
- **Fad check.** Each topic's share of papers over time, so booms and fades are easy to see.
- **Research briefing.** A short written and audio summary of what's moving.

## Run it

Python 3.12.

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python html_main.py             # open http://localhost:8000
```

The site reads the committed files in `data/`, so it works offline from a fresh clone. With a `GEMINI_API_KEY` in `.env`, it also fetches new arXiv papers and tags them every 2 hours while it runs. Set `LIVE_REFRESH=0` to turn that off.

Run the tests with `pytest`. Rebuild every data file with `python main.py` (this is slow and calls live APIs).

## Tech stack

- **Data:** OpenAlex (10,232 papers, 2000–2026) and the arXiv q-fin feed
- **Models:** SPECTER2 (Hugging Face `transformers`), scikit-learn, Gemini (`google-genai`), ElevenLabs for the audio
- **Site:** a plain Python HTTP server with D3.js, no framework

## Models

1. **Topics.** Each paper's title and abstract is turned into a vector with SPECTER2, a model pre-trained on scientific papers. PCA and a Gaussian mixture group the vectors into 25 topics, which we named by hand. The map shows the 8 largest. The fitted groups are frozen in `data/group_model.npz`, so new papers can be placed in the same topics.
2. **Growth forecast.** A ridge regression ranks topics by how much their share of papers will grow over the next three years. Its seven features describe each topic's recent history: past growth, acceleration, current share, volatility, links to other topics, how spread out its papers are, and how cleanly they fit the topic.
3. **arXiv tagging.** Gemini places each new arXiv paper into one of the topics. This is what feeds the monthly view.

## Evaluation

- **Growth forecast:** tested walk-forward from 2010 to 2022, so each year's forecast only uses earlier papers, and the topics are refitted each time so nothing leaks from the future. The rank correlation with actual growth was 0.30 (random scores 0). Of its top 5 picks, an average of 1.6 really were top-5 growers, against 1.0 for random picks. It also beat simply following past growth (0.30 against 0.15), but that gap was not significant after correcting for multiple tests (p = 0.07). So we show the forecast as exploratory, not proven.
- **Gemini tags:** 46 of 50 randomly sampled papers (92%) were tagged correctly when checked by hand.

## Layout

```
html_main.py          the website (server and page)
static/               CSS and JS (radar.js: topic drill-down, fads.js: Fad check)
src/research_map/     embeddings, topic groups, citation network
src/topic_radar/      growth forecast and its tests
src/live/             arXiv feed, Gemini tagging, briefing
data/                 every file the site reads
```
