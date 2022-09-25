# snoop2head's blog

### Commands

**testing purpose on the local**

```bash
npm run build
npm run start
```

**updates gh-pages branch and publishes**

```bash
npm run deploy
git add .
git commit -m "message"
git push
```

```json
{
  "build": "gatsby build",
  "develop": "gatsby develop",
  "start": "npm run develop",
  "deploy": "gatsby build && gh-pages -d public"
}
```

**Move previous posts to current blog**

```bash
python migrate.py
```

### Sources

- Fork of [JaeSeoKim's Blog](https://github.com/JaeSeoKim/jaeseokim.github.io)
- [Configuring a publishing source for your GitHub Pages site](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)
- [How to host a Gatsby Website on Github Pages](https://www.youtube.com/watch?v=8tz9zDmrEbA&t=303s)
