import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import Index from "./pages/Index"; // Corrected import
import NotFound from "./pages/NotFound"; // Ensure NotFound.tsx exists
import Analysis from "./pages/Analysis"; // Ensure Analysis.tsx exists

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Index />} />
        <Route path="/analysis" element={<Analysis />} />
        <Route path="*" element={<NotFound />} />
      </Routes>
    </Router>
  );
}

export default App; // Single default export