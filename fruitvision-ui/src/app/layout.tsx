import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "FruitVision AI",
  description: "Premium fruit quality detection dashboard for SMS presentation."
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
