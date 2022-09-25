import React, { useState } from "react"
import { graphql } from "gatsby"
import Markdown from "../components/Markdown"
import Layout from "../components/Layout"
import Divider from "../components/Divider"
import Tags from "../components/Tags"
import tw from "twin.macro"
const Wrapper = tw.div`w-full max-w-screen-md mx-auto`

export default ({ data }) => {
  const resumes = data.allMarkdownRemark.edges
  const [currentHeaderUrl, setCurrentHeaderUrl] = useState(undefined)

  const resume = resumes
    .filter((resume) => resume.node.frontmatter.title === "Resume")
    .map(({ node }) => node)[0]

  return (
    <>
      <Layout>
        <div css={tw`mt-4 px-4`} className="blog-post-container">
          <div className="blog-post">
            <Wrapper>
              <h1
                className="blog-title"
                css={tw`text-4xl md:text-5xl font-bold mb-4`}
              >
                {"Portfolio"}
              </h1>
              <h2 className="blog-date" css={tw`text-base mb-4`}>
                {"September 25th 2022"}
              </h2>

              <Divider color />
            </Wrapper>
          </div>
        </div>
        <div css={tw`mt-4 px-4`} className={"blog-content"}>
          <Wrapper>
            <Markdown html={resume.html} />
          </Wrapper>
        </div>
      </Layout>
    </>
  )
}

export const query = graphql`
  query {
    allMarkdownRemark(
      sort: { fields: [frontmatter___date], order: DESC }
      filter: { frontmatter: { draft: { eq: true } } }
    ) {
      edges {
        node {
          excerpt(pruneLength: 200, truncate: true)
          html
          fields {
            slug
          }
          frontmatter {
            date(formatString: "YYYY-MM-DD")
            title
            draft
          }
        }
      }
    }
  }
`
