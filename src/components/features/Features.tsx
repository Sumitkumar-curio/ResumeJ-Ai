import { Button } from "@/components/ui/button";
import { 
  Zap, 
  Target, 
  Brain, 
  Shield, 
  BarChart3, 
  FileCheck,
  Clock,
  Users,
  Award
} from "lucide-react";

const features = [
  {
    icon: Target,
    title: "Match Score Optimization",
    description: "Achieve 90%+ Match Score compatibility with intelligent keyword matching and format optimization.",
    color: "primary"
  },
  {
    icon: Brain,
    title: "AI-Powered Analysis",
    description: "Advanced AI analyzes your resume against job descriptions for personalized recommendations.",
    color: "secondary"
  },
  {
    icon: BarChart3,
    title: "Real-time Scoring",
    description: "Get instant feedback with detailed scoring across multiple resume quality metrics.",
    color: "accent"
  },
  
];

const stats = [
  {
    icon: Users,
    value: "50K+",
    label: "Happy Users",
    description: "Professionals trust ResumeJ AI"
  },
  {
    icon: Target,
    value: "95%",
    label: "Success Rate",
    description: "Users see improved response rates"
  },
  {
    icon: Zap,
    value: "10x",
    label: "Faster Hiring",
    description: "Average time to job offer"
  },
  {
    icon: Shield,
    value: "100%",
    label: "Privacy Secure",
    description: "Your data is always protected"
  }
];

export default function Features() {
  return (
    <section id="features" className="py-24 px-4 lg:px-8 relative">
      <div className="max-w-7xl mx-auto">
        {/* Section Header */}
        <div className="text-center space-y-6 mb-16">
          <div className="inline-flex items-center gap-2 glass-card text-sm text-secondary font-medium">
            <Zap className="w-4 h-4" />
            <span>Why Choose ResumeJ AI</span>
          </div>
          
          <h2 className="text-3xl sm:text-4xl lg:text-6xl font-bold">
            <span className="text-gradient-primary">Powerful Features</span>
            <br />
            for Resume Success
          </h2>
          
          <p className="text-lg sm:text-xl text-muted-foreground max-w-3xl mx-auto">
            Our AI-powered platform combines advanced technology with industry expertise 
            to give your resume the competitive edge it needs.
          </p>
        </div>

        {/* Features Grid */}
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6 lg:gap-8 mb-20">
          {features.map((feature, index) => {
            const Icon = feature.icon;
            return (
              <div 
                key={index} 
                className="glass-card hover:scale-105 transition-all duration-300 group cursor-pointer"
              >
                <div className={`inline-flex items-center justify-center w-12 h-12 rounded-xl bg-gradient-${feature.color} mb-6 glow-${feature.color} group-hover:scale-110 transition-transform`}>
                  <Icon className="w-6 h-6 text-white" />
                </div>
                
                <h3 className="text-xl font-semibold mb-3 text-foreground">
                  {feature.title}
                </h3>
                
                <p className="text-muted-foreground leading-relaxed">
                  {feature.description}
                </p>
              </div>
            );
          })}
        </div>

        {/* Stats Section */}
        <div className="glass rounded-2xl p-8 lg:p-12">
          <div className="text-center mb-12">
            <h3 className="text-3xl lg:text-4xl font-bold mb-4">
              Proven Results You Can <span className="text-gradient-accent">Trust</span>
            </h3>
            <p className="text-lg text-muted-foreground">
              Join thousands of professionals who've transformed their careers with ResumeJ AI
            </p>
          </div>
          
          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-8">
            {stats.map((stat, index) => {
              const Icon = stat.icon;
              return (
                <div key={index} className="text-center">
                  <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-primary glow-primary mb-4 animate-pulse-glow">
                    <Icon className="w-8 h-8 text-white" />
                  </div>
                  <div className="text-3xl lg:text-4xl font-bold text-gradient-primary mb-2">
                    {stat.value}
                  </div>
                  <div className="font-semibold text-foreground mb-1">
                    {stat.label}
                  </div>
                  <div className="text-sm text-muted-foreground">
                    {stat.description}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* CTA */}
        <div className="text-center mt-16">
          <Button variant="hero" size="lg" className="text-lg px-12 py-6">
            Start Your Free Analysis
          </Button>
        </div>
      </div>
    </section>
  );
}