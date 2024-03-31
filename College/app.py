from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# Load dataset
df = pd.read_csv('data.csv')

# Train recommendation model
model = NearestNeighbors(n_neighbors=20)
model.fit(df[['Rank', 'Intregrated Rank', 'Gender', 'Region', 'Caste Code', 'Branch Code']])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    rank = int(data['rank'])
    integrated_rank = int(data['integrated_rank'])
    gender = data['gender']
    region = data['region']
    caste_code = data['caste_code']
    branch_code = data['branch_code']

    _, indices = model.kneighbors([[rank, integrated_rank, gender, region, caste_code, branch_code]])
    recommended_colleges = df.iloc[indices[0]][['Inst_ code', 'InstitutionName', 'BranchCode', 'Rank', 'Intregrated Rank', 'Gender', 'Region', 'Caste Code']].to_dict(orient='records')

    return jsonify(recommended_colleges)

if __name__ == '__main__':
    app.run(debug=True)
