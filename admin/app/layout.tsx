import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "KalapakTTS — Admin",
  description: "Administrator console for KalapakTTS.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full">{children}</body>
    </html>
  );
}
