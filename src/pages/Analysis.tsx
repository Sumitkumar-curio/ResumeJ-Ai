import React, { useEffect, useState } from 'react';
import { useLocation } from 'react-router-dom';

const Analysis: React.FC = () => {
  const location = useLocation();
  const [analysisResult, setAnalysisResult] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const formData = location.state?.formData as FormData;
    if (formData) {
      fetch('http://localhost:8000/analyze', {
        method: 'POST',
        body: formData,
      })
        .then(response => {
          if (!response.ok) {
            throw new Error('Failed to fetch analysis results');
          }
          return response.json();
        })
        .then(data => {
          setAnalysisResult(data);
          setLoading(false);
        })
        .catch(error => {
          setError(error.message);
          setLoading(false);
        });
    }
  }, [location.state]);

  if (loading) {
    return <div>Loading...</div>;
  }

  if (error) {
    return <div>Error: {error}</div>;
  }

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-3xl font-bold mb-4">Resume Analysis Results</h1>
      {analysisResult && (
        <div>
          <h2 className="text-2xl font-semibold">Overall Score: {analysisResult.score}</h2>
          <div className="my-4">
            <h3 className="text-xl font-semibold">Full Analysis</h3>
            <p>{analysisResult.full_analysis}</p>
          </div>
          <div className="my-4">
            <h3 className="text-xl font-semibold">Suggestions</h3>
            <ul>
              {analysisResult.suggestions.map((suggestion: string, index: number) => (
                <li key={index}>{suggestion}</li>
              ))}
            </ul>
          </div>
          <div className="my-4">
            <h3 className="text-xl font-semibold">Visualizations</h3>
            <div className="flex space-x-4">
              {analysisResult.visualizations.map((viz: string, index: number) => (
                <img key={index} src={`data:image/png;base64,${viz}`} alt="Analysis visualization" />
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Analysis;
