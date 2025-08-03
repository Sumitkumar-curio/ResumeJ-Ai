import { Button } from "@/components/ui/button";
import { Upload, Sparkles, Target } from "lucide-react";
import heroImage from "@/assets/hero-image.jpg";

export default function Hero() {
  return (
    <section id="home" className="relative min-h-screen flex items-center justify-center px-4 lg:px-8 pt-20">
      <div className="max-w-7xl mx-auto grid lg:grid-cols-2 gap-12 items-center">
        {/* Left Content */}
        <div className="text-center lg:text-left space-y-8 z-10">
          <div className="space-y-4">
            <div className="inline-flex items-center gap-2 glass-card text-sm text-primary font-medium">
              <Sparkles className="w-4 h-4" />
              <span>AI-Powered Resume Optimization</span>
            </div>
            
            <h1 className="text-4xl sm:text-5xl lg:text-7xl font-bold leading-tight">
              Optimize Your{" "}
              <span className="text-gradient-primary">Resume</span>
              <br />
              Beat the{" "}
              <span className="text-gradient-accent">Match Score</span>
            </h1>
            
            <p className="text-lg sm:text-xl text-muted-foreground max-w-2xl mx-auto lg:mx-0">
              Transform your resume with AI-powered insights. Get 85%+ Match Scores, 
              personalized feedback, and land your dream job faster.
            </p>
          </div>

          <div className="flex flex-col sm:flex-row gap-4 justify-center lg:justify-start">
            <Button variant="hero" size="lg" className="text-lg px-8 py-6">
              <Upload className="w-5 h-5" />
              Analyze My Resume
            </Button>
            <Button variant="outline" size="lg" className="text-lg px-8 py-6">
              <Target className="w-5 h-5" />
              See Demo
            </Button>
          </div>

          <div className="flex flex-wrap items-center justify-center lg:justify-start gap-6 sm:gap-8 text-sm text-muted-foreground">
            <div className="text-center">
              <div className="text-2xl font-bold text-gradient-primary">95%</div>
              <div>Success Rate</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-gradient-secondary">10K+</div>
              <div>Resumes Optimized</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-gradient-accent">24/7</div>
              <div>AI Support</div>
            </div>
          </div>
        </div>

        {/* Right Content - Hero Image */}
        <div className="relative order-first lg:order-last">
          <div className="relative animate-float">
            <div className="absolute inset-0 bg-gradient-primary rounded-2xl lg:rounded-3xl blur-2xl opacity-20"></div>
            <img 
              src={heroImage} 
              alt="ResumeJ AI Dashboard" 
              className="relative z-10 w-full h-auto rounded-2xl lg:rounded-3xl shadow-2xl border border-white/10"
            />
          </div>
          
          {/* Floating Stats */}
          {/* <div className="absolute top-4 sm:top-10 -left-2 sm:-left-6 glass-card animate-pulse-glow text-xs sm:text-sm">
            <div className="text-center p-2 sm:p-4">
              <div className="text-lg sm:text-2xl font-bold text-gradient-primary">92%</div>
              <div className="text-xs sm:text-sm text-muted-foreground">Match Score</div>
            </div>
          </div> */}
          
          {/* <div className="absolute bottom-8 sm:bottom-20 -right-2 sm:-right-8 glass-card animate-pulse-glow text-xs sm:text-sm">
            <div className="text-center p-2 sm:p-4">
              <div className="text-lg sm:text-2xl font-bold text-gradient-accent">8.5/10</div>
              <div className="text-xs sm:text-sm text-muted-foreground">Match Score</div>
            </div>
          </div> */}
        </div>
      </div>
    </section>
  );
}