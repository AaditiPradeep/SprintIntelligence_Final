"""Local API server for the Sprint Intelligence dashboard."""
import json
import sys
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; FRONTEND=ROOT/'dist'
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
_recommender = None
REQUIRED={'issue_key','volatility_risk','resolution_risk','reopen_risk','combined_risk','risk_band','planner_action'}
def probability(value,name):
    try: value=float(value)
    except (TypeError,ValueError) as exc: raise ValueError(f'{name} must be numeric.') from exc
    if not 0<=value<=1: raise ValueError(f'{name} must be between 0 and 1.')
    return value
def summarize(items):
    if not isinstance(items,list) or not items: raise ValueError('Upload a non-empty risk register.')
    missing=REQUIRED-set(items[0])
    if missing: raise ValueError('This is not a Sprint Intelligence risk register. Missing: '+', '.join(sorted(missing)))
    clean=[]
    for row in items:
        row=dict(row)
        for name in ('volatility_risk','resolution_risk','reopen_risk','combined_risk'): row[name]=probability(row.get(name),name)
        clean.append(row)
    clean.sort(key=lambda row:row['combined_risk'],reverse=True); mean=sum(row['combined_risk'] for row in clean)/len(clean); high=sum(row['combined_risk']>=.70 for row in clean); score=round(100*(1-(.70*mean+.30*high/len(clean))),1)
    return {'candidate_item_count':len(clean),'mean_combined_risk':round(mean,4),'high_risk_item_count':high,'sprint_health_score_out_of_100':score,'sprint_health_band':'Healthy' if score>=70 else 'At Risk' if score>=45 else 'Critical','action_counts':{a:sum(row.get('planner_action')==a for row in clean) for a in ('Defer','Split','Monitor','Include')},'items':clean}

def top_factors(row, risk_type):
    """Read optional per-item XAI factors, with a truthful score-only fallback."""
    risk_keys = {'requirement_volatility': 'volatility_risk', 'resolution_risk': 'resolution_risk', 'reopen_risk': 'reopen_risk'}
    explanations = row.get('explanations', {})
    if isinstance(explanations, dict):
        factors = explanations.get(risk_keys[risk_type], {}).get('factors', [])
        if factors:
            result = []
            for factor in factors[:5]:
                contribution = factor.get("contribution")
                if isinstance(contribution, (int, float)):
                    result.append(
                        f"{factor['label']} {factor['direction']} the model score relative "
                        f"to its reference (probability change {contribution:+.3f})"
                    )
                else:
                    result.append(f"{factor['label']} {factor['direction']} the model score relative to its reference")
            return result
    raw = row.get('top_factors', '')
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            value = json.loads(raw)
            if isinstance(value, list):
                return [str(item) for item in value]
        except json.JSONDecodeError:
            pass
        return [part.strip() for part in raw.split(';') if part.strip()]
    return [f"Elevated predicted {risk_type.replace('_', ' ')} probability."]

def add_llm_recommendations(items):
    """Generate approved-strategy recommendations lazily, after user request."""
    global _recommender
    if len(items) > 30:
        raise ValueError('Generate recommendations for at most 30 sprint items at a time.')
    from llm.recommender import MitigationRecommender, RiskInput
    if _recommender is None:
        _recommender = MitigationRecommender()
    labels = {
        'volatility_risk': 'requirement_volatility',
        'resolution_risk': 'resolution_risk',
        'reopen_risk': 'reopen_risk',
    }
    enriched = []
    for item in items:
        row = dict(item)
        highest = max(labels, key=lambda key: float(row[key]))
        risk_type = labels[highest]
        risk = RiskInput(
            issue_key=row['issue_key'],
            risk_type=risk_type,
            risk_probability=float(row[highest]),
            top_factors=top_factors(row, risk_type),
            planner_action=row['planner_action'],
            issue_summary=str(row.get('summary', row.get('summary_at_cutoff', ''))),
        )
        row['llm_recommendation'] = _recommender.generate(risk)
        prediction_key = {
            'requirement_volatility': 'volatility_risk',
            'resolution_risk': 'resolution_risk',
            'reopen_risk': 'reopen_risk',
        }[risk_type]
        row['llm_recommendation']['model_context'] = {
            'selected_prediction_task': risk_type,
            'prediction_model': row.get('resolution_model', 'selected classical model'),
            'explanation_method': row.get('explanations', {}).get(prediction_key, {}).get('method', ''),
        }
        enriched.append(row)
    return enriched
class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*args,**kwargs): super().__init__(*args,directory=str(FRONTEND),**kwargs)
    def end_headers(self):
        # Local development dashboard: always serve the current frontend after
        # a refresh rather than retaining an earlier JavaScript bundle.
        self.send_header('Cache-Control', 'no-store, max-age=0')
        super().end_headers()
    def do_GET(self):
        if self.path=='/api/health': return self.reply(HTTPStatus.OK,{'status':'ok'})
        super().do_GET()
    def do_POST(self):
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=5_000_000: raise ValueError('Upload a risk register smaller than 5 MB.')
            request=json.loads(self.rfile.read(length).decode())
            items=request.get('items')
            if self.path=='/api/score-candidates':
                from dashboard.backend.explain import score_candidates
                return self.reply(HTTPStatus.OK, summarize(score_candidates(items)))
            if self.path=='/api/sprint-summary':
                return self.reply(HTTPStatus.OK,summarize(items))
            if self.path=='/api/llm-recommendations':
                return self.reply(HTTPStatus.OK,{'items':add_llm_recommendations(items)})
            if self.path=='/api/optimize-sprint':
                from dashboard.backend.optimizer import optimise_sprint
                return self.reply(HTTPStatus.OK, optimise_sprint(items, request.get('capacity')))
            return self.reply(HTTPStatus.NOT_FOUND,{'error':'Unknown API endpoint.'})
        except (ValueError,json.JSONDecodeError,ImportError,RuntimeError) as exc:
            self.reply(HTTPStatus.BAD_REQUEST,{'error':str(exc)})
    def reply(self,status,payload):
        """Send JSON without treating a cancelled browser request as a server crash."""
        body=json.dumps(payload).encode()
        try:
            self.send_response(status)
            self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Content-Length',str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            # The page was refreshed, navigated away, or cancelled an in-flight
            # request. The scoring result is simply no longer needed by that tab.
            pass
if __name__=='__main__':
    print('Sprint Intelligence dashboard: http://127.0.0.1:8000');ThreadingHTTPServer(('127.0.0.1',8000),Handler).serve_forever()
