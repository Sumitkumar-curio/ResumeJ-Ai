import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Github, Twitter, Linkedin, Mail, Heart } from "lucide-react";

export default function Footer() {
  const currentYear = new Date().getFullYear();

  return (
    <footer id="contact" className="py-16 px-4 lg:px-8 border-t border-border/50">
      <div className="max-w-7xl mx-auto">
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8 mb-12">
          {/* Brand */}
          <div className="space-y-4">
            <h3 className="text-2xl font-bold text-gradient-primary">ResumeJ AI</h3>
            <p className="text-muted-foreground">
              AI-powered resume optimization platform helping professionals land their dream jobs with 95% success rate.
            </p>
            <div className="flex gap-3">
            <a href="https://t.me/Resumejobai" target="_blank" rel="noopener noreferrer">
            <Button variant="ghost" size="icon" className="hover:bg-primary/20">
              <Twitter className="w-4 h-4" />
            </Button>
            </a>
            <a href="https://www.linkedin.com/company/resumej-ai" target="_blank" rel="noopener noreferrer">
            <Button variant="ghost" size="icon" className="hover:bg-primary/20">
              <Linkedin className="w-4 h-4" />
            </Button>
             </a>
            <Button variant="ghost" size="icon" className="hover:bg-primary/20">
            <Github className="w-4 h-4" />
            </Button>
            <a href="mailto:sumitkumar.kr1248@gmail.com">
            <Button variant="ghost" size="icon" className="hover:bg-primary/20">
            <Mail className="w-4 h-4" />
            </Button>
          </a>
  
            </div>
          </div>

          {/* Product */}
          <div className="space-y-4">
            <h4 className="font-semibold text-foreground">Product</h4>
            <ul className="space-y-2 text-muted-foreground">
              <li><a href="#demo" className="hover:text-primary transition-colors">Try Demo</a></li>
              <li><a href="#features" className="hover:text-primary transition-colors">Features</a></li>
              <li><a href="#pricing" className="hover:text-primary transition-colors">Pricing</a></li>
              <li><a href="#api" className="hover:text-primary transition-colors">API</a></li>
            </ul>
          </div>

          {/* Resources */}
          <div className="space-y-4">
            <h4 className="font-semibold text-foreground">Resources</h4>
            <ul className="space-y-2 text-muted-foreground">
              <li><a href="#blog" className="hover:text-primary transition-colors">Blog</a></li>
              <li><a href="#guides" className="hover:text-primary transition-colors">Resume Guides</a></li>
              <li><a href="#templates" className="hover:text-primary transition-colors">Templates</a></li>
              <li><a href="#help" className="hover:text-primary transition-colors">Help Center</a></li>
            </ul>
          </div>

          {/* Company */}
          <div className="space-y-4">
            <h4 className="font-semibold text-foreground">Company</h4>
            <ul className="space-y-2 text-muted-foreground">
              <li><a href="#about" className="hover:text-primary transition-colors">About Us</a></li>
              <li><a href="#careers" className="hover:text-primary transition-colors">Careers</a></li>
              <li><a href="#contact" className="hover:text-primary transition-colors">Contact</a></li>
              <li><a href="#press" className="hover:text-primary transition-colors">Press Kit</a></li>
            </ul>
          </div>
        </div>

        <Separator className="mb-8 bg-border/50" />

        <div className="flex flex-col md:flex-row justify-between items-center gap-4">
          <div className="text-muted-foreground text-sm">
            © {currentYear} ResumeJ AI. All rights reserved.
          </div>
          
          <div className="flex items-center gap-6 text-sm text-muted-foreground">
            <a href="#privacy" className="hover:text-primary transition-colors">Privacy Policy</a>
            <a href="#terms" className="hover:text-primary transition-colors">Terms of Service</a>
            <a href="#cookies" className="hover:text-primary transition-colors">Cookie Policy</a>
          </div>
          
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            {/* <span>Made at </span> */}
            {/* <Heart className="w-4 h-4 text-red-500 fill-current animate-pulse" /> */}
            {/* <span>by </span> */}
          </div>
        </div>
      </div>
    </footer>
  );
}