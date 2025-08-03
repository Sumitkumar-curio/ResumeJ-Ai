import { Star, Quote } from "lucide-react";

const testimonials = [
  {
    name: "Rajesh",
    role: "Software Engineer",
    content: "ResumeJ AI helped me increase my Match Score from 67% to 94%. I got 3x more interview calls and landed my dream job at Google!",
    rating: 5,
    avatar: "👩‍💻"
  },
  {
    name: "Maya Singh", 
    role: "Product Manager ",
    content: "The AI insights were incredibly detailed. Within 2 weeks of optimizing my resume, I had offers from 3 major tech companies.",
    rating: 5,
    avatar: "👨‍💼"
  },
  // {
  //   name: "Emily Rodriguez",
  //   role: "Marketing Director at Tesla", 
  //   content: "I was struggling to get past initial screenings. ResumeJ AI's optimization got me a 40% response rate improvement immediately.",
  //   rating: 5,
  //   avatar: "👩‍🎨"
  // },
  // {
  //   name: "David Park",
  //   role: "Data Scientist at Netflix",
  //   content: "The keyword optimization and formatting suggestions were spot-on. My resume now passes every Match Score system effortlessly.",
  //   rating: 5,
  //   avatar: "👨‍🔬"
  // },
  // {
  //   name: "Lisa Thompson",
  //   role: "UX Designer at Apple",
  //   content: "From 15% to 89% Match Score in just one session. The personalized feedback helped me understand exactly what recruiters look for.",
  //   rating: 5,
  //   avatar: "👩‍🎨"
  // },
  // {
  //   name: "James Wilson",
  //   role: "Finance Manager at JPMorgan",
  //   content: "I tried other tools but none came close to ResumeJ AI's accuracy. The AI analysis is incredibly sophisticated and actionable.",
  //   rating: 5,
  //   avatar: "👨‍💼"
  // }
];

export default function Testimonials() {
  return (
    <section id="reviews" className="py-24 px-4 lg:px-8 relative">
      <div className="max-w-7xl mx-auto">
        {/* Section Header */}
        <div className="text-center space-y-6 mb-16">
          <div className="inline-flex items-center gap-2 glass-card text-sm text-secondary font-medium">
            <Star className="w-4 h-4" />
            <span>Success Stories</span>
          </div>
          
          <h2 className="text-3xl sm:text-4xl lg:text-6xl font-bold">
            What Our Users Are <span className="text-gradient-secondary">Saying</span>
          </h2>
          
          <p className="text-lg sm:text-xl text-muted-foreground max-w-3xl mx-auto">
            Join thousands of professionals who've transformed their careers with ResumeJ AI
          </p>
        </div>

        {/* Testimonials Grid */}
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6 lg:gap-8">
          {testimonials.map((testimonial, index) => (
            <div 
              key={index}
              className="glass-card hover:scale-105 transition-all duration-300 group relative overflow-hidden"
            >
              {/* Quote Icon */}
              <div className="absolute top-4 right-4 opacity-10 group-hover:opacity-20 transition-opacity">
                <Quote className="w-8 h-8 text-primary" />
              </div>
              
              {/* Rating */}
              <div className="flex items-center gap-1 mb-4">
                {[...Array(testimonial.rating)].map((_, i) => (
                  <Star key={i} className="w-4 h-4 fill-accent text-accent" />
                ))}
              </div>
              
              {/* Content */}
              <p className="text-muted-foreground mb-6 leading-relaxed">
                "{testimonial.content}"
              </p>
              
              {/* Author */}
              <div className="flex items-center gap-3">
                <div className="w-12 h-12 rounded-full bg-gradient-primary flex items-center justify-center text-2xl glow-primary">
                  {testimonial.avatar}
                </div>
                <div>
                  <div className="font-semibold text-foreground">
                    {testimonial.name}
                  </div>
                  <div className="text-sm text-muted-foreground">
                    {testimonial.role}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Stats */}
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-8 mt-16 text-center">
          <div className="glass-card">
            <div className="text-3xl lg:text-4xl font-bold text-gradient-primary mb-2">
              4.9/5
            </div>
            <div className="text-muted-foreground">Average Rating</div>
          </div>
          <div className="glass-card">
            <div className="text-3xl lg:text-4xl font-bold text-gradient-secondary mb-2">
              50K+
            </div>
            <div className="text-muted-foreground">Happy Users</div>
          </div>
          <div className="glass-card">
            <div className="text-3xl lg:text-4xl font-bold text-gradient-accent mb-2">
              95%
            </div>
            <div className="text-muted-foreground">Success Rate</div>
          </div>
          <div className="glass-card">
            <div className="text-3xl lg:text-4xl font-bold text-gradient-primary mb-2">
              3x
            </div>
            <div className="text-muted-foreground">More Interviews</div>
          </div>
        </div>
      </div>
    </section>
  );
}