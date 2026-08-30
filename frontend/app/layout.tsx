import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Extractive Question Answering | SQuAD 1.1",
  description:
    "Extractive Question Answering over user-supplied passages using Transformer " +
    "start/end span prediction, fine-tuned on SQuAD 1.1.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-950 text-slate-100 antialiased">
        {children}
      </body>
    </html>
  );
}
