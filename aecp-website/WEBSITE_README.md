# AECP Website

## 📦 Complete Static Website Package

A fully-functional, zero-cost-to-host static website for AECP with:
- Interactive documentation
- Performance benchmarks with live charts
- Client-side playground (no backend needed!)
- Cost calculator
- Protocol visualization

## 🎯 Zero Server Costs

**Everything runs client-side:**
- No backend required
- No database needed
- No API calls to your servers
- Pure HTML/CSS/JavaScript
- Can be hosted on GitHub Pages, Netlify, Vercel (all free!)

## 📁 File Structure

```
aecp-website/
├── index.html              # Home page with hero, stats, quick start
├── protocol.html           # How AECP works (4 phases explained)
├── performance.html        # Benchmarks, charts, cost calculator
├── playground.html         # Interactive demo (client-side)
├── docs.html               # Full API documentation
├── css/
│   └── style.css          # Complete styling
├── js/
│   ├── main.js            # Main site JavaScript
│   ├── protocol-charts.js # Protocol visualization
│   ├── performance-charts.js # Performance charts
│   ├── playground.js      # Interactive playground
│   └── visualization.js   # Transfer visualization
├── data/
│   └── benchmarks.json    # Benchmark data for charts
└── images/
    └── (add logos, screenshots as needed)
```

## 🚀 Quick Start

### Option 1: GitHub Pages (Recommended - FREE)

1. Create a new repository: `aecp-website`
2. Upload all files
3. Go to Settings → Pages
4. Select `main` branch and `/` root
5. Your site will be live at: `https://yourusername.github.io/aecp-website`

### Option 2: Netlify (FREE)

1. Drag and drop the `aecp-website` folder to Netlify
2. Done! Site is live instantly
3. Get a custom domain (optional)

### Option 3: Vercel (FREE)

```bash
cd aecp-website
vercel --prod
```

### Option 4: Local Preview

```bash
cd aecp-website
python -m http.server 8000
# Visit: http://localhost:8000
```

## 📊 Features

### Home Page (`index.html`)
- Hero section with key metrics
- Code examples (Python & TypeScript)
- Problem/solution explanation
- Quick links
- Responsive design

### Protocol Page (`protocol.html`)
- 4-phase explanation with visuals
- Technical deep dive
- Math formulas
- Comparison table
- Interactive charts

### Performance Page (`performance.html`)
- Live benchmarks
- Interactive cost calculator
- Scalability charts
- Model compatibility matrix
- Real-world use cases
- Benchmark methodology

### Playground Page (`playground.html`)
- **Client-side demo** - No backend needed!
- Mock embedding generation
- Transfer simulation
- Similarity visualization
- Cost estimation
- All computations in browser

### Docs Page (`docs.html`)
- Complete API reference
- Installation instructions
- Code examples
- Provider guides
- Plugin system docs

## 🎨 Customization

### Update Branding

Edit these in `css/style.css`:
```css
:root {
  --primary-color: #6366f1;     /* Main brand color */
  --secondary-color: #8b5cf6;   /* Accent color */
  --success-color: #10b981;     /* Success/positive */
  --danger-color: #ef4444;      /* Warning/negative */
}
```

### Update Content

1. **GitHub Links:** Search and replace `yourusername` with your actual username
2. **Package Names:** Update PyPI/npm links if different
3. **Metrics:** Update numbers in `performance.html` with your benchmarks
4. **Examples:** Customize code examples in `index.html`

### Add Analytics

Add before `</head>` in all HTML files:
```html
<!-- Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-XXXXXXXXXX"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-XXXXXXXXXX');
</script>
```

## 📈 Performance

The website is optimized for speed:
- Minimal dependencies (only Chart.js)
- No heavy frameworks
- Lazy loading for charts
- Optimized CSS
- Fast load times (&lt;2s)

## 🔒 Security

All computation happens client-side:
- No sensitive data sent to servers
- No user data collection
- No cookies required
- Privacy-friendly

## 🌐 Browser Support

Works on all modern browsers:
- Chrome/Edge (2020+)
- Firefox (2020+)
- Safari (2020+)
- Mobile browsers

## 📝 Maintenance

### Updating Benchmarks

Edit `data/benchmarks.json`:
```json
{
  "semantic_similarity": {
    "text": 0.43,
    "aecp": 0.86
  },
  "latency_ms": {
    "text": 150,
    "aecp": 0.8
  }
}
```

### Adding New Pages

1. Copy an existing HTML file
2. Update navigation links
3. Add to footer
4. Update sitemap

## 🎯 Next Steps

1. **Deploy:** Choose a hosting option above
2. **Customize:** Update branding and content
3. **Share:** Add link to README.md
4. **Monitor:** Add analytics if desired
5. **Maintain:** Keep benchmarks updated

## 💡 Tips

1. **Images:** Add screenshots to `images/` folder
2. **SEO:** Update meta descriptions in each HTML file
3. **Social:** Add Open Graph tags for better sharing
4. **Mobile:** Test on mobile devices (already responsive)
5. **Speed:** Use CDN for faster loads globally

## 🐛 Troubleshooting

### Charts not showing?
- Check browser console for errors
- Ensure Chart.js CDN is accessible
- Verify JavaScript files are loaded

### Styles broken?
- Check CSS file path
- Clear browser cache
- Verify font-awesome CDN

### Links not working?
- Use relative paths for local testing
- Update absolute URLs for production

## 📞 Support

- **Issues:** GitHub Issues
- **Docs:** See `docs.html`
- **Examples:** See playground

## License

MIT License - Same as AECP
