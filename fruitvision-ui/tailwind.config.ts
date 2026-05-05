import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/lib/**/*.{js,ts,jsx,tsx,mdx}"
  ],
  theme: {
    extend: {
      colors: {
        ink: "#111714",
        mist: "#f5f7f5",
        leaf: "#1f7a4d",
        limewash: "#e9f6ed"
      },
      boxShadow: {
        premium: "0 24px 70px rgba(17, 23, 20, 0.11)",
        soft: "0 14px 40px rgba(17, 23, 20, 0.08)"
      }
    }
  },
  plugins: []
};

export default config;
