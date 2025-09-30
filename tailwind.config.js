/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx,js,jsx}"],
  theme: {
    extend: {
      colors: {
        // Palette discrète pour l’app
        brand: {
          900: "#0f172a",
          700: "#1f2937",
          500: "#334155",
          300: "#94a3b8",
        },
      },
    },
  },
  plugins: [],
}
