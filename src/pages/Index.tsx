import { useState } from "react";
import ParticleBackground from "@/components/ParticleBackground";
import Navigation from "@/components/Navigation";
import Hero from "@/components/Hero";
import Features from "@/components/features/Features";
import DemoForm from "@/components/features/DemoForm";
import AnalysisModal from "@/components/features/AnalysisModal";
import Testimonials from "@/components/features/Testimonials";
import Footer from "@/components/Footer";

const Index = () => {
  const [analysisResults, setAnalysisResults] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  const handleAnalysisComplete = (results: any) => {
    setAnalysisResults(results);
    setIsModalOpen(true);
  };

  return (
    <div className="min-h-screen bg-gradient-hero relative overflow-hidden">
      {/* Particle Background - Temporarily commented out due to WebGL issues */}
      {/* <ParticleBackground /> */}
      
      {/* Navigation */}
      <Navigation />
      
      {/* Main Content */}
      <div className="relative z-10">
        <Hero />
        <DemoForm setResults={handleAnalysisComplete} />
        <Features />
        
        <AnalysisModal results={analysisResults} isOpen={isModalOpen} setIsOpen={setIsModalOpen} />
        <Testimonials />
        <Footer />
      </div>
    </div>
  );
};


export default Index;
