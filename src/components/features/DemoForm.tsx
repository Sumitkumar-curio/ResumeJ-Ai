import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Upload, Sparkles, FileText, Target } from "lucide-react";
import { useToast } from "@/components/ui/use-toast";

interface DemoFormProps {
  setResults: (results: any) => void;
}

const DemoForm = ({ setResults }: DemoFormProps) => {
  const [file, setFile] = useState<File | null>(null);
  const [jobDescription, setJobDescription] = useState("");
  const [selectedProfile, setSelectedProfile] = useState("");
  const [finalJobDescription, setFinalJobDescription] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [jobProfilesData, setJobProfilesData] = useState<Array<{ name: string; description: string }>>([]);
  const { toast } = useToast();

  useEffect(() => {
    if (selectedProfile) {
      const profile = jobProfilesData.find(p => p.name === selectedProfile);
      if (profile) {
        setFinalJobDescription(profile.description);
      }
    } else {
      setFinalJobDescription(jobDescription);
    }
  }, [jobDescription, selectedProfile, jobProfilesData]);

  useEffect(() => {
    const fetchJobProfiles = async () => {
      try {
        const response = await fetch("http://localhost:8000/job-profiles");
        if (!response.ok) {
          throw new Error("Failed to fetch job profiles.");
        }
        const data = await response.json();
        console.log("Fetched job profiles:", data.profiles); // Log fetched data
        setJobProfilesData(data.profiles);
      } catch (error) {
        console.error("Error fetching job profiles:", error); // Log error
        toast({
          title: "Error fetching job profiles",
          description: "Could not load predefined job profiles.",
          variant: "destructive",
        });
      }
    };
    fetchJobProfiles();
  }, [toast]);

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const uploadedFile = event.target.files?.[0];
    if (uploadedFile) {
      if (uploadedFile.size > 5 * 1024 * 1024) {
        toast({
          title: "File too large",
          description: "Please upload a file smaller than 5MB.",
          variant: "destructive",
        });
        return;
      }
      
      if (!uploadedFile.type.includes('pdf') && !uploadedFile.type.includes('doc') && !uploadedFile.type.includes('docx')) {
        toast({
          title: "Invalid file type",
          description: "Please upload a PDF or DOCX file.",
          variant: "destructive",
        });
        return;
      }
      
      setFile(uploadedFile);
      toast({
        title: "File uploaded successfully",
        description: `${uploadedFile.name} is ready for analysis.`,
      });
    }
  };

  const handleAnalyze = async () => {
    if (!file) {
      toast({
        title: "Missing resume",
        description: "Please upload your resume first.",
        variant: "destructive",
      });
      return;
    }

    if (!finalJobDescription) {
      toast({
        title: "Missing job information",
        description: "Please provide a job description or select a profile.",
        variant: "destructive",
      });
      return;
    }

    setIsAnalyzing(true);
    if(setResults) setResults(null);

    const formData = new FormData();
    formData.append("resume", file);
    formData.append("job_description", finalJobDescription);

    try {
      const response = await fetch("http://localhost:8000/analyze", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: "Failed to analyze resume." }));
        throw new Error(errorData.detail);
      }

      const data = await response.json();
      
      if(setResults) setResults(data);

    } catch (error: any) {
      console.error("Analysis error:", error); // Log the full error for debugging
      toast({
        title: "Analysis failed",
        description: error.message || "Could not connect to the backend.",
        variant: "destructive",
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <section className="py-24 px-4 lg:px-8 relative" id="resume-analysis">
      <div className="max-w-4xl mx-auto">
        <div className="text-center space-y-6 mb-12">
          <div className="inline-flex items-center gap-2 glass-card text-sm text-accent font-medium">
            <Sparkles className="w-4 h-4" />
            <span>Free Demo - Try Now</span>
          </div>
          
          <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold">
            See ResumeJ AI in <span className="text-gradient-accent">Action</span>
          </h2>
          
          <p className="text-lg sm:text-xl text-muted-foreground">
            Upload your resume and get instant feedback on how to improve your Match Score
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-6 lg:gap-8">
          {/* Input Form */}
          <Card className="glass border-border/50">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Upload className="w-5 h-5 text-primary" />
                Upload & Analyze
              </CardTitle>
              <CardDescription>
                Get your personalized resume analysis in seconds
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* File Upload */}
              <div className="space-y-2">
                <Label htmlFor="resume">Resume (PDF or DOCX)</Label>
                <div className="relative">
                  <Input
                    id="resume"
                    type="file"
                    accept=".pdf,.doc,.docx"
                    onChange={handleFileUpload}
                    className="file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-medium file:bg-primary file:text-primary-foreground hover:file:bg-primary/90"
                  />
                  {file && (
                    <div className="mt-2 text-sm text-green-400 flex items-center gap-2">
                      <FileText className="w-4 h-4" />
                      {file.name}
                    </div>
                  )}
                </div>
              </div>

              {/* Job Profile Selector */}
              <div className="space-y-2">
                <Label htmlFor="profile">Job Profile</Label>
                <p className="text-sm text-muted-foreground mb-2">
                  Option 2: Select a predefined job profile for a general analysis.
                </p>
                <Select value={selectedProfile} onValueChange={setSelectedProfile}>
                  <SelectTrigger className="glass border-border/50">
                    <SelectValue placeholder="Select your target role" />
                  </SelectTrigger>
                  <SelectContent className="glass border-border/50 bg-card/95 backdrop-blur-xl z-50">
                    {jobProfilesData.map((profile) => (
                      <SelectItem key={profile.name} value={profile.name}>
                        {profile.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Job Description */}
              <div className="space-y-2">
                <Label htmlFor="description">
                  Job Description <span className="text-muted-foreground">(Optional, if you selected a Job Profile above)</span>
                </Label>
                <p className="text-sm text-muted-foreground mb-2">
                  Option 1: Paste a specific job description for a tailored analysis.
                </p>
                <Textarea
                  id="description"
                  placeholder="Paste the job description here for a highly accurate match score..."
                  value={jobDescription}
                  onChange={(e) => setJobDescription(e.target.value)}
                  className="glass border-border/50 min-h-[120px] resize-none"
                  maxLength={10000}
                />
                <div className="text-xs text-muted-foreground text-right">
                  {jobDescription.length}/10000
                </div>
              </div>

              <Button 
                onClick={handleAnalyze}
                disabled={isAnalyzing}
                className="w-full"
                variant="hero"
                size="lg"
              >
                {isAnalyzing ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Target className="w-5 h-5 mr-2" />
                    Analyze Resume
                  </>
                )}
              </Button>
            </CardContent>
          </Card>
        </div>
      </div>
    </section>
  );
};

export default DemoForm;