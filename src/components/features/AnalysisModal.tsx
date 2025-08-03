
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";

const ScoreCircle = ({ score }: { score: number }) => {
  const circumference = 2 * Math.PI * 40;
  const offset = circumference - (score / 100) * circumference;

  return (
    <div className="relative inline-flex items-center justify-center overflow-hidden rounded-full w-28 h-28">
      <svg className="w-full h-full">
        <circle className="text-gray-300" strokeWidth="8" stroke="currentColor" fill="transparent" r="40" cx="56" cy="56" />
        <circle
          className="text-primary"
          strokeWidth="8"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          stroke="currentColor"
          fill="transparent"
          r="40"
          cx="56"
          cy="56"
          style={{ transition: 'stroke-dashoffset 0.5s ease-in-out' }}
        />
      </svg>
      <span className="absolute text-2xl font-bold text-primary">{score.toFixed(1)}%</span>
    </div>
  );
};

const AnalysisModal = ({ results, isOpen, setIsOpen }: { results: any, isOpen: boolean, setIsOpen: (isOpen: boolean) => void }) => {
  if (!results) {
    return null;
  }

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogContent className="max-w-3xl h-[80vh] md:h-[90vh] glass border-border/50 bg-card/95 backdrop-blur-xl flex flex-col">
        <DialogHeader>
          <DialogTitle className="text-2xl">Resume Analysis Report</DialogTitle>
          <DialogDescription>Here's a detailed breakdown of your resume's match score.</DialogDescription>
        </DialogHeader>
        <Tabs defaultValue="overview" className="w-full flex-grow flex flex-col">
          <TabsList className="grid w-full grid-cols-2 sm:grid-cols-4">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="keywords">Keywords</TabsTrigger>
            <TabsTrigger value="suggestions">Suggestions</TabsTrigger>
            <TabsTrigger value="charts">Visuals</TabsTrigger>
          </TabsList>
          <ScrollArea className="flex-grow mt-4 pr-4">
            <TabsContent value="overview">
              <Card className="bg-transparent border-none">
                <CardHeader className="text-center">
                  <CardTitle>Overall Match Score</CardTitle>
                </CardHeader>
                <CardContent className="flex flex-col items-center justify-center gap-4">
                  <ScoreCircle score={results.score} />
                  <div className="text-center">
                      <p className="text-lg">Your resume is a <strong>{results.score.toFixed(1)}%</strong> match for this job.</p>
                      <p className="text-muted-foreground">Experience detected: {results.experience_years.toFixed(1)} years</p>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
            <TabsContent value="keywords">
               <Card className="bg-transparent border-none">
                  <CardHeader>
                      <CardTitle>Keyword Analysis</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                      <div>
                          <h4 className="font-semibold text-lg mb-2">Matched Keywords</h4>
                          <div className="flex flex-wrap gap-2">
                              {results.keyword_matches.matched_keywords.map((keyword: string) => (
                                  <Badge key={keyword} variant="default">{keyword}</Badge>
                              ))}
                          </div>
                      </div>
                      <div>
                          <h4 className="font-semibold text-lg mb-2">Missing Keywords</h4>
                          <div className="flex flex-wrap gap-2">
                              {results.keyword_matches.missing_keywords.map((keyword: string) => (
                                  <Badge key={keyword} variant="destructive">{keyword}</Badge>
                              ))}
                          </div>
                      </div>
                      <div>
                          <h4 className="font-semibold text-lg mb-2">No Match Keywords</h4>
                          <div className="flex flex-wrap gap-2">
                              {results.keyword_matches.no_match_keywords.map((keyword: string) => (
                                  <Badge key={keyword} variant="secondary">{keyword}</Badge>
                              ))}
                          </div>
                      </div>
                  </CardContent>
               </Card>
            </TabsContent>
            <TabsContent value="suggestions">
              <Card className="bg-transparent border-none">
                  <CardHeader>
                      <CardTitle>Improvement Suggestions</CardTitle>
                  </CardHeader>
                  <CardContent>
                      <ul className="list-disc pl-5 space-y-2">
                          {results.suggestions.map((suggestion: string) => <li key={suggestion}>{suggestion}</li>)}
                      </ul>
                  </CardContent>
              </Card>
            </TabsContent>
            <TabsContent value="charts">
               <Card className="bg-transparent border-none">
                  <CardHeader>
                      <CardTitle>Visual Breakdown</CardTitle>
                  </CardHeader>
                  <CardContent className="flex flex-col sm:flex-row justify-center items-center gap-4">
                      {results.visualizations.map((viz: string, index: number) => (
                      <img key={index} src={`data:image/png;base64,${viz}`} alt={`Visualization ${index + 1}`} className="rounded-lg shadow-lg max-w-full sm:max-w-xs"/>
                      ))}
                  </CardContent>
              </Card>
            </TabsContent>
          </ScrollArea>
        </Tabs>
      </DialogContent>
    </Dialog>
  );
};

export default AnalysisModal;
