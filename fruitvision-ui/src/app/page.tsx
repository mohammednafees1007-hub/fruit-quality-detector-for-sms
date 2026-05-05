import { AIExplanation } from "@/components/AIExplanation";
import { Dashboard } from "@/components/Dashboard";
import { DetectionPanel } from "@/components/DetectionPanel";
import { Footer } from "@/components/Footer";
import { Gallery } from "@/components/Gallery";
import { Hero } from "@/components/Hero";
import { HowItWorks } from "@/components/HowItWorks";
import { Navbar } from "@/components/Navbar";

export default function Home() {
  return (
    <main className="overflow-hidden">
      <Navbar />
      <Hero />
      <HowItWorks />
      <DetectionPanel />
      <Dashboard />
      <Gallery />
      <AIExplanation />
      <Footer />
    </main>
  );
}
