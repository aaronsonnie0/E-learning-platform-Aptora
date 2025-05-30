
import React from "react";
import { TeamMember } from "./TeamMember";

// Import team member photos
import aaronPhoto from "../assets/aaron.jpg";
import kariPhoto from "../assets/kari.jpg";
import sonaPhoto from "../assets/sona.jpg";
import thrithwikPhoto from "../assets/thrithwik.jpg";

export const OurTeam = () => {
  const teamMembers = [
    {
      name: "Aaron Sonnie",
      role: "Founder & Backend Developer",
      bio: "Passionate backend developer with expertise in building scalable, efficient, and secure systems.",
      photoUrl: "/lovable-uploads/7ac96ef4-2820-482a-ae2e-a05176205147.png",
      linkedin: "https://linkedin.com/in/aaronsonnie",
      github: "https://github.com/aaronsonnie",
      email: "aaron@aptora.com"
    },
    {
      name: "Karivaradhan",
      role: "Founder & TeamLead",
      bio: "Passionate about creating innovative educational solutions. Leads the technical development and strategic direction of Aptora.",
      photoUrl: "/lovable-uploads/b94a9ece-7e3f-42eb-bc23-a5eb21a13b71.png",
      linkedin: "https://linkedin.com/in/karivaradhan",
      github: "https://github.com/karivaradhan",
      email: "kari@aptora.com"
    },
    {
      name: "Sona Daison",
      role: "Founder & Frontend Developer",
      bio: "Creates beautiful, intuitive interfaces that make learning engaging and accessible for all users.",
      photoUrl: "/lovable-uploads/0486d9ae-e6fa-4ba8-89e7-2a8e9e37b657.png",
      linkedin: "https://linkedin.com/in/sonadaison",
      github: "https://github.com/sonadaison",
      email: "sona@aptora.com"
    },
    {
      name: "Thrithwik",
      role: "Co-Founder & Developer",
      bio: "Develops the robust infrastructure that powers Aptora's AI learning capabilities and data processing.",
      photoUrl: thrithwikPhoto,
      linkedin: "https://linkedin.com/in/thrithwik",
      github: "https://github.com/thrithwik",
      email: "thrithwik@aptora.com"
    }
  ];

  return (
    <section id="team" className="py-16">
      <div className="container px-4 md:px-6">
        <div className="text-center mb-12">
          <h2 className="text-3xl md:text-4xl font-bold tracking-tight mb-4">
            Our Team
          </h2>
          <p className="text-xl text-muted-foreground max-w-[700px] mx-auto">
            Meet the talented individuals behind Aptora's mission to transform education with AI.
          </p>
        </div>
        
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-8">
          {teamMembers.map((member) => (
            <TeamMember key={member.name} {...member} />
          ))}
        </div>
      </div>
    </section>
  );
};
