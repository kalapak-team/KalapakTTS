import type { Metadata } from "next";
import type { ReactNode } from "react";
import "./globals.css";

export const metadata: Metadata = {
  title: "KalapakTTS — Khmer Voice Studio",
  description: "Offline Khmer text-to-speech studio for this PC.",
};

export default function RootLayout({ children }: Readonly<{ children: ReactNode }>) {
  return (
    <html lang="km">
      <body>{children}</body>
    </html>
  );
}
