<xsl:stylesheet version="1.0"
xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
<xsl:output method="text" encoding="iso-8859-1"/>
<xsl:strip-space elements="*" />

<xsl:template match="/testsuites">  
  <xsl:apply-templates select="testsuite" />
</xsl:template>

<xsl:template match="/testsuites/testsuite">  
  <xsl:apply-templates select="testcase" />
</xsl:template>

<xsl:template match="/testsuites/testsuite/testcase">
  <xsl:value-of select="../../@name"/><xsl:text>&#09;</xsl:text>
  <xsl:value-of select="../@name"/><xsl:text>&#09;</xsl:text>
  <xsl:value-of select="@name"/><xsl:text>&#09;</xsl:text>
  <xsl:value-of select="@status"/><xsl:text>&#09;</xsl:text>
  <xsl:text>&#10;</xsl:text>
</xsl:template>

</xsl:stylesheet>
