from modules.chain import Chain
from tqdm import tqdm
import pandas as pd
from services.sheets_service import get_sheet_data_and_df, save_to_sheet
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit as st
from services.helper_functions import validate_column_values, get_outline_with_los
from services.smart_progress_bar import SmartProgressBar


extract_relevant_info_from_article_as_lo_prompt = """
<examples>\n<example>\n<COURSE_NAME>\nTaking Temperature Measurements\n</COURSE_NAME>\n<SEARCH_QUERY>\ndigital thermometers for HVAC applications\n</SEARCH_QUERY>\n<ROUGH_OUTLINE>\nTopic: Thermometers\n  Subtopic: Digital thermometers\n  Subtopic: Temperature probes\n  Subtopic: Infrared thermometers & thermal imaging\n  Subtopic: Gauge / meter calibration\n  Subtopic: Dry bulb and wet bulb Delta T\n</ROUGH_OUTLINE>\n<ARTICLE_CONTENT>\nUS Only\nWe only ship within the United States\nYour Zip:\nPlumbing supplies, heating supplies, HVAC supplies from SupplyHouse Homepage\nSearch for a product, brand, or SKU…\n\nHelp Face\nCONTACT & SUPPORT\nLive Chat Offline: Email Us\nCart0\nCART\nPlumbing\nHeating\nHVAC\nElectrical\nPEX\nFittings\nValves\nTools\nResources\nOur Team\nApply to be a TradeMaster!\nMagnifying glass\nHover image to zoom\nHomeHVAC SuppliesToolsUEi Test InstrumentsThermometers\nPDT550, Digital NSF Pocket Thermometer\nBrand:\nUEi Test Instruments\nSKU:\nPDT550\n\n(2)\nQ&A:\n(0)\n$23.82 each\nIn Stock\nIn Stock\nGet 31 Fri, Nov 22\nMANUALS (1)\nProduct Overview\nFree Shipping On orders over $99\nEasy Returns No restocking fee for 90 days\nDescription\n–\nThe PDT550 is a versatile and rugged pocket thermometer featuring an extended temperature range, built-in magnetic mount and waterproof case. The PDT550 provides reliable and accurate measurements in demanding environments. They are perfect for HVAC, refrigeration and food service areas, or anywhere temperature monitoring is critical.\n\nFeatures:\nMeasures from:58 to +572?F (-50 to +300?C)\nWaterproof\nFive translucent colors for unique applications*:\nSmoke\nGoldenrod\nEmerald\nRoyal Blue\nCherry\nBuilt in magnetic mount\nRecords minimum and maximum temperature\nHOLD freezes the current temperature reading\nEasy access battery compartment\nEnhanced accuracy of digital circuitry\nIncludes battery, and multipurpose probe cover\nThree year limited warranty\nSpecs\n–\nProduct Type:\nDigital Pocket Thermometer\n\nContact\nHelp Face\nQuestion? Call or text 888-757-4774\n+\nPDT550, Digital NSF Pocket Thermometer\nPDT550, Digital NSF Pocket Thermometer\nBrand:\nUEi Test Instruments\nSKU:\nPDT550\n\n(2)\nQ&A:\n(0)\nIn Stock\n$23.82 each\nProduct Info\nYou May Also Need (1)\nReviews (2)\nQ&A (0)\nYou May Also Need\n2 Product Reviews\nWrite a Review\n5.0\n100%\nof respondents would recommend this product\n5 Stars\t\n2\t\n4 Stars\t\t0\t\n3 Stars\t\t0\t\n2 Stars\t\t0\t\n1 Star\t\t0\t\nEnter Keywords to Search Reviews\n\n\nMost Recent\nReviewed by 2 customers\nGood product at a good price!\nSubmitted 4 years ago\n\nBy Mr. Fixit\n\nFrom Goldsboro, N.C.\n\nVerified Buyer\n\nComments about PDT550, Digital NSF Pocket Thermometer\n\nWe use these thermometer at our YMCA to record pool temperatures for our pools and for our spas. They are durable and accurate.\n\nMore Details\t\nBottom Line Yes, I would recommend to a friend\n\nWas this review helpful to you?\n\t1\t0\nFlag this review\nGood product for the $\nSubmitted 4 years ago\n\nBy KHVAC\n\nFrom Sycamore, IL\n\nVerified Buyer\n\nComments about PDT550, Digital NSF Pocket Thermometer\n\nAlways hold up to daily use for a couple of yrs\n\nMore Details\t\nBottom Line Yes, I would recommend to a friend\n\nWas this review helpful to you?\n\t1\t0\nFlag this review\nDisplaying Reviews 1-2\n\nProduct Q&A\nAsk a Question\nFree Shipping\nOn orders over $99\nWe offer flexible shipping and scheduling options, up-to-date delivery estimates, and free ground shipping on any order over $99 to make sure your order gets to you on-time, for less.\nEasy Returns\nReturning items is as simple as 1-2-3\nOrdered the wrong product? Item damaged in transit? Our 3-step process makes returning products easy – even up to a year after purchase – with no restocking fees on returns made within 90 days.\nWe're Here To Help!\nEmail\nResponse by Thurs\nCall or Text 888-757-4774\nFooter People Non TradeMaster\nOrder Status\nContact Us\nShipping Policy\nReturns\nFAQs\nMeet The Team\nCareers\nOur Blog\nReorder\nQuick Order\nSaved Carts\nProject Calculators\nVideo Library\nTradeMaster Program\nTradeMaster Spotlight\nTrade Life\nGive Us Feedback\nWe'd love to hear about your experience with us.\nSubscribe for More\nStay in the know with top brands & product spotlights!\nYour email address\nApp Image\nDownload Our App!\nEasier and faster shopping on our 200,000+ products.  Learn More\nGoogle Play App Store\nApple App Store\nSupplyHouse Youtube Page\nSupplyHouse X Page\nSupplyHouse Facebook Page\nSupplyHouse Instagram Page Page\nSupplyHouse Linkedin Page Page\nSupplyHouse TikTok Page Page\n© 2024 SupplyHouse.com\nTerms of Use\nPrivacy Policy\nAccessibility\nSitemap\nDo Not Sell My Personal Information\nGoogle Seller\nBBB Reliability Seal\nReseller Ratings\nPHCC\n\n</ARTICLE_CONTENT>\n<TARGET_AUDIENCE>\nEntry-level HVAC technicians\n</TARGET_AUDIENCE>\n<USER_GUIDELINES>\nNone.\n</USER_GUIDELINES>\n<COURSE_BACKGROUND>\nCourse related information:\n- This course on temperature measurements is part of a larger learning plan on Airflow and Ductwork.\n- This will be a self-paced e-learning course.\n- The target audience for this course is entry-level technicians.\n</COURSE_BACKGROUND>\n<ideal_output>\n<summary> The article is a product listing for the UEi PDT550 digital pocket thermometer, featuring technical specifications, product features, and customer reviews. It describes a waterproof thermometer with a temperature range of -58 to +572°F, magnetic mount capability, and various color options. The listing includes basic product information and two positive customer reviews discussing its use in pool maintenance and HVAC applications. </summary> <analysis_breakdown> Let me analyze this article systematically: 1. Content Overview: - The article is primarily a product listing for the PDT550 digital pocket thermometer - Contains technical specifications, features, and customer reviews - Focused on selling rather than education - Limited technical explanation of concepts 2. Alignment with Course Topics: - Relates to \"Digital thermometers\" subtopic - Contains basic temperature measurement range information - Lacks detailed explanation of measurement principles - No coverage of other course topics (calibration, Delta T, etc.) 3. Target Audience Considerations: - Entry-level HVAC technicians need foundational knowledge - Product specs alone don't provide necessary educational context - Missing important conceptual information about temperature measurement - Customer reviews suggest practical applications but don't explain methodology 4. Educational Value Assessment: - Primarily marketing content - Features list could inform discussion of thermometer capabilities - Lacks explanatory content needed for learning - No troubleshooting or best practices information </analysis_breakdown> <content_evaluation> Key passages: 1. \"Measures from -58 to +572°F (-50 to +300°C)\" - Relates to temperature measurement ranges 2. \"Records minimum and maximum temperature\" - Connects to basic measurement functionality Potential connections to course objectives: - Temperature measurement range capabilities - Basic digital thermometer features - Real-world applications (from reviews) Complexity relative to target audience: Too basic and product-focused; lacks educational depth needed for entry-level technicians Content type: Product-focused Overall relevance: While the article relates to temperature measurement tools, it is primarily a product listing lacking the educational content necessary for a comprehensive learning experience. The information is too superficial and marketing-oriented to serve as core educational material. This article is not suitable for creating comprehensive learning objectives because it focuses on product specifications rather than explaining concepts, principles, or proper measurement techniques that entry-level HVAC technicians need to understand. </content_evaluation> <objective_brainstorm> While the article isn't suitable for comprehensive learning objectives, it could contribute to limited technical awareness objectives: - Understanding basic digital thermometer features - Recognizing temperature measurement ranges - Identifying basic maintenance considerations </objective_brainstorm> <final_verdict> Note: Given the product-focused nature of the content, I recommend not using this article as a primary source for learning objectives. The content would be better suited as a supplementary reference when discussing types of measurement tools available in the field. Any learning objectives created from this content alone would be too limited in scope and depth for meaningful educational outcomes for entry-level HVAC technicians. If this article were to be used, it should be combined with additional educational materials that explain the underlying concepts, proper measurement techniques, and practical applications in HVAC systems. Skip adding LOs from this article </final_verdict> <learning_objectives> NONE </learning_objectives>\n</ideal_output>\n</example>\n<example>\n<COURSE_NAME>\nTaking Temperature Measurements\n</COURSE_NAME>\n<SEARCH_QUERY>\ndigital thermometers for HVAC applications\n</SEARCH_QUERY>\n<ROUGH_OUTLINE>\nTopic: Thermometers\n  Subtopic: Digital thermometers\n  Subtopic: Temperature probes\n  Subtopic: Infrared thermometers & thermal imaging\n  Subtopic: Gauge / meter calibration\n  Subtopic: Dry bulb and wet bulb Delta T\n</ROUGH_OUTLINE>\n<ARTICLE_CONTENT>\nSkip to content\nApply To HVAC Jobs\n\nPost HVAC Employer Job\n\nblue air conditioning\nHVAC Tactician\nYour Guide To HVAC Parts & Repair\nSearch\nSearch\nSearch...\nShop Now\nInfo Menu\nMenu\nTypes of Thermometers Used in HVAC\nTypes of Thermometers Used in HVAC\nOctober 10, 2023\nRobert \"Bob\" Mitchell\nMeasurement & Testing\nHome > HVAC Tools > Measurement & Testing > Types of Thermometers Used in HVAC\n\nTable of Contents\nWhen it comes to working on HVAC systems, accurate temperature measurement is crucial. Different types of thermometers are used in the HVAC industry to ensure precise temperature readings. These thermometers are designed to measure temperature in various ways, depending on the specific needs of an HVAC technician. In this article, we will discuss the different types of thermometers commonly used in HVAC.\n\n1. Digital Thermometers\nProbe Thermometers: Probe thermometers consist of a metal needle-like probe that is inserted into the substance or component being measured. They provide accurate and quick temperature readings and are ideal for measuring temperatures in hard-to-reach areas, such as inside ducts or pipes.\nInfrared Thermometers: Infrared thermometers use infrared radiation to measure temperature. They are non-contact thermometers that can measure temperatures from a distance. These thermometers are useful for quickly assessing surface temperatures, such as air vents or heating coils.\nThermocouple Thermometers: Thermocouple thermometers utilize the principle of thermoelectric effect to measure temperature. They consist of two different metal wires, called thermocouples, that generate a voltage when exposed to heat. These thermometers are highly accurate and commonly used for HVAC testing and troubleshooting.\nResistance Temperature Detector (RTD) Thermometers: RTD thermometers use the electrical resistance of metals to measure temperature. They provide highly accurate and reliable readings and are often used in HVAC systems that require precise temperature control.\n2. Analog Thermometers\nBimetallic Thermometers: Bimetallic thermometers use a coil made of two different metals with different coefficients of thermal expansion. When the temperature changes, the coil expands or contracts, providing a reading on the temperature scale. These thermometers are commonly used in HVAC systems for temperature monitoring.\nLiquid-Filled Thermometers: Liquid-filled thermometers contain a liquid, often mercury or alcohol, that expands or contracts with temperature changes. The movement of the liquid is then used to indicate the temperature. These thermometers are simple and inexpensive, making them suitable for general temperature measurement in HVAC systems.\n3. Wireless Thermometers\nWireless thermometers eliminate the need for physical contact with the thermometer itself. Instead, they send temperature measurements wirelessly to a receiver or display unit. These thermometers are ideal for remote monitoring of temperatures in HVAC systems or for measuring temperatures in hazardous or hard-to-reach areas.\n\nHow to Choose the Right Thermometer for HVAC\nWhen selecting a thermometer for HVAC use, consider the following factors:\n\nAccuracy: Choose a thermometer with a high level of accuracy to ensure precise temperature readings. Look for thermometers that have been tested and certified for accuracy.\nRange: Consider the temperature range that the thermometer can measure. Ensure that it covers the anticipated temperature range in your HVAC system.\nFeatures: Look for additional features that may be useful for your specific HVAC needs, such as data logging, adjustable emissivity in infrared thermometers, or compatibility with wireless systems.\nDurability: HVAC systems can be harsh environments, so select a thermometer that is built to withstand the conditions it will be exposed to.\nFrequently Asked Questions (FAQs)\nQ1: Are digital thermometers more accurate than analog thermometers?\n\nA1: Generally, digital thermometers offer higher accuracy than analog thermometers. Digital thermometers provide precise numerical readings, while analog thermometers rely on visual interpretation of a scale. However, accuracy can vary depending on the specific model and calibration of the thermometer.\n\nQ2: Can I use a household thermometer for HVAC measurements?\n\nA2: It is not recommended to use a household thermometer for HVAC measurements. Household thermometers are usually not calibrated for the temperature ranges and precision required in HVAC systems. It’s best to use thermometers specifically designed for HVAC applications.\n\nQ3: Do I need multiple types of thermometers for HVAC work?\n\nA3: It can be beneficial to have multiple types of thermometers for HVAC work, as different thermometers excel at measuring specific things. For instance, a probe thermometer is great for measuring temperature inside ducts, while an infrared thermometer is ideal for measuring surface temperatures. Having a variety of thermometers on hand allows for more accurate and efficient temperature measurements.\n\nQ4: Can I use a wireless thermometer for HVAC system monitoring?\n\nA4: Yes, wireless thermometers are excellent for HVAC system monitoring. They allow for remote temperature monitoring from a central location, making them ideal for large HVAC systems or when monitoring multiple locations simultaneously.\n\nQ5: How often should I calibrate my thermometer?\n\nA5: It is recommended to calibrate your thermometer at least once a year or according to the manufacturer’s guidelines. Calibration ensures that the thermometer is providing accurate temperature readings and helps maintain the overall efficiency and performance of your HVAC system.\n\nIn conclusion, thermometers play a crucial role in the HVAC industry for measuring temperature accurately. Understanding the different types of thermometers and their specific applications can help HVAC technicians choose the right tools for their work. Whether it’s digital, analog, or wireless, selecting the appropriate thermometer ensures precise temperature measurements and efficient HVAC system operation.\n\nHVAC Business Software\n\nHousecall Pro\nServiceWorks \nHVAC Jobs\n\nPost A Job\nApply To Job\n\nMastering HVAC Performance: The Essential Guide to Pressure Measurement Tools\nRead More »\n\nMastering HVAC Humidity Control: Essential Tools and Techniques\nRead More »\n\nEssential HVAC Electrical Testing: Tools and Measurement Mastery\nRead More »\n\nHVAC Duct Leakage Testing: Essential Tools and Procedures\nRead More »\n\nMastering HVAC: Essential Tools and Measurement Techniques to Avoid Common Errors\nRead More »\n\nMastering HVAC Maintenance: Utilize Ultrasonic Tools for Precise Leak Detection\nRead More »\nMastering Climate Control: Tracing the Evolution of HVAC Measurement Tools\nRead More »\nRelated Posts\n\nMastering HVAC Performance: The Essential Guide to Pressure Measurement Tools\nMaximize HVAC efficiency with proper pressure measurement – a key to system maintenance and energy savings. Understand the significance of pressure in HVAC\n\nRead More\n\nMastering HVAC Humidity Control: Essential Tools and Techniques\nExplore essential tools like hygrometers, humidifiers, and dehumidifiers alongside techniques such as proper ventilation and regular HVAC maintenance to master indoor humidity control.\n\nRead More\n\nEssential HVAC Electrical Testing: Tools and Measurement Mastery\nIn this informative post on HVAC electrical testing, you’ll learn about the importance of using the correct tools, such as multimeters and ammeters,\n\nRead More\n\nHVAC Duct Leakage Testing: Essential Tools and Procedures\nHVAC duct leakage testing is a critical maintenance task that ensures system efficiency, reduces energy costs, and prolongs the lifespan of your HVAC\n\nRead More\n\nMastering HVAC: Essential Tools and Measurement Techniques to Avoid Common Errors\nDive into the crucial aspects of HVAC mastery with our comprehensive guide covering the pivotal tools and measurement techniques needed to excel in\n\nRead More\n\nMastering HVAC Maintenance: Utilize Ultrasonic Tools for Precise Leak Detection\nDiscover how ultrasonic tools are revolutionizing HVAC maintenance, offering precise and rapid leak detection that surpasses conventional methods. Ultrasonic detectors identify high-frequency sounds\n\nRead More\nMastering Climate Control: Tracing the Evolution of HVAC Measurement Tools\nDiscover the transformation of HVAC measurement tools that keep our indoor environments comfortable and efficient. From the earliest thermometers to the high-tech smart\n\nRead More\nEssential HVAC Testing and Measurement Tools for Beginners: A Comprehensive Guide\nDiscover the must-have HVAC testing and measurement tools for beginners in this comprehensive guide. Learn about the essential instruments, like thermometers, pressure gauges,\n\nRead More\nThe Impact of Accurate Measurement on HVAC Efficiency\nMaster the efficiency of your HVAC system with precise measurement and testing tools. In this comprehensive guide, uncover the crucial role of tools\n\nRead More\nMaximizing Your HVAC System’s Performance with Precise Measurement Tools\nMaximize your HVAC system’s efficiency with our guide on the essential tools and techniques for precise measurement and testing. Discover the critical role\n\nRead More\nEssential Toolkit: The Top 10 HVAC Measurement and Testing Tools Every Professional Needs\nDiscover the top 10 HVAC measurement and testing tools critical for professionals in the industry, including a manifold gauge set, multimeter, refrigerant scale,\n\nRead More\nEssential HVAC Measurement and Testing Instruments: A Professional’s Top 10 Toolkit\nDiscover the essential tools every HVAC professional needs with our expert roundup of the top 10 measurement and testing instruments, from multimeters and\n\nRead More\n\nRevolutionizing HVAC Diagnostics: How Advanced Measurement Tools Solve Complicated Problems – A Collection of Case Studies\nDiscover the breakthroughs in HVAC diagnostics with our in-depth exploration of advanced measurement tools. We delve into real-world case studies that exemplify how\n\nRead More\nCategories\nCarrier Parts\nCarrier Circuit Boards\nCarrier Heat Exchangers\nCarrier Miscellaneous Parts\nCarrier Motors & Motor Parts\nCarrier Pilots & Ignition\nCarrier Sensors\nCarrier Switches\nCarrier Valves\nCooling\nAC & Furnace Systems\nAccessories\nAir Conditioner Condensers\nAir Conditioning Systems\nAir Handlers\nCommercial Packaged Units\nHeat Pump & Furnace Systems\nHeat Pump Condensers\nHeat Pump Systems\nMini-Split Systems\nMobile Home Units\nPackaged Units\nPortable, Room & Window AC Units\nPTAC Units\nSolar Heat Pump Systems\nDuctless Mini Split\n2 Zone 2 Room Mini-Split Systems\n3 Zone 3 Room Mini-Split Systems\n4 Zone 4 Room Mini-Split Systems\n5 Zone 5 Room Mini-Split Systems\n6 Zone 6 Room Mini-Split Systems\n7 and 8 Zone Mini-Split Systems\nDIY Mini-Split Systems\nDuctless Mini-Split Accessories\nSingle Zone 1 Room Mini-Split Systems\nElectrical & Solar\nBackup Batteries\nElectrical Accessories\nPower Stations\nSolar Accessories\nSolar Panels\nSolar-Powered Air Conditioning Systems\nEnterprise\nCompliance and Legal Affairs\nCorporate Strategy and Governance\nCustomer Relationship Management (CRM)\nEnterprise Sales and Marketing\nFinancial Planning and Analysis\nIndustry Knowledge\nLarge-scale Project Management\nNational and International Operations\nQuality Assurance and Control\nResearch and Development\nRisk Management and Insurance\nSupply Chain Management\nSustainability and Environmental Compliance\nTalent Acquisition and Workforce Development\nTechnology Integration and Automation\nVendor and Partner Relationships\nFans & Ventilation\nAgricultural Fans\nAir Circulation Fans\nAir Quality ERV HRV IAQ\nCommercial & Industrial Wall-Mounted Exhaust Fans\nCommercial Ducting\nExhaust Fans\nHazardous Location & Corrosion-Resistant Fans\nHome Ventilation & Exhaust Fans\nHVLS Fans\nIndustrial Fans\nInline Fans\nRestaurant Fans and Ventilation\nRoof Exhaust Fans\nUtility Set Fans\nFire & Outdoor\nBBQ Accessories\nBBQ Grills\nChimney Products\nCooking & Heating Stoves\nFire Pits & Tables\nFireplace Accessories\nIndoor Fireplaces\nOutdoor Fireplaces\nOutdoor Kitchens\nPatio Heaters\nGoodman Parts\nGoodman Blowers\nGoodman Control Boards\nGoodman Miscellaneous Parts\nGoodman Motors & Motor Parts\nGoodman Pilots & Ignition\nGoodman Propane Conversion Kits\nGoodman Sensors\nGoodman Switches\nGoodman Valves\nHeating\nCommercial Units\nElectric Heating\nFurnace & AC Systems\nFurnaces\nFurnaces & Coil Combinations\nFurnaces and Coil Systems\nInfrared Heaters\nUnit Heaters\nWater Heaters & Boilers\nHiring HVAC Tech\nAdditional Services\nCost and Financing\nElectrical Insulation Mats\nEnvironmental Considerations\nExperience and Expertise\nHow To Hire HVAC Technician\nInsurance and Liability\nLicensing and Certification\nLocation and Availability\nProfessionalism and Customer Service\nReferences\nReviews and Testimonials\nWarranties and Guarantees\nWhen To Hire HVAC Technician\nWhere To Hire HVAC Technician\nHVAC Tools\nCleaning & Maintenance\nCutting & Shaping\nInstallation & Assembly\nMeasurement & Testing\nSafety Tools\nIndependent Tech Career\nBusiness Operations\nCertification and Licensing\nClient Referrals\nCommunity Involvement\nDirect Outreach\nFinancial Planning and Benefits\nGetting Started\nJob Estimation and Bidding\nLocal Advertising\nMarketing and Client Acquisition\nNetworking and Industry Involvement\nOngoing Education and Skill Development\nOnline Service Marketplaces\nPartnerships with Local Businesses\nSafety and Compliance\nService Excellence and Reputation Building\nSocial Media Presence\nTools and Equipment\nWebsite and Online Marketing\nIndustry\nSoftware Reviews\nStats\nTrends\nMobile Home\nHeat & A/C Package Units for Mobile Homes\nHeat Pumps for Mobile Homes\nManufactured and Mobile Home Furnaces\nManufactured and Mobile Home HVAC System Accessories\nManufactured Home Heat Pump Systems\nMobile Home Air Conditioning and Coil Systems\nMobile Home Air Conditioning and Furnace Systems\nMobile Home Air Conditioning Units\nMobile Home Evaporator Coils\nMobile Home Heat Pump & Evaporator Coil Systems\nModine Parts\nModine Control Boards\nModine Miscellaneous Parts\nModine Motors & Motor Parts\nModine Switches\nReznor Parts\nReznor Fan Blades\nReznor Miscellaneous Parts\nReznor Motors & Motor Parts\nReznor Switches\nSmall Business\nBusiness Planning\nCustomer Service and Engagement\nEmergency and After-Hours Services\nEquipment and Inventory Management\nFinancial Management\nMarketing and Sales\nOperations and Workflow\nQuality Control\nRegulatory Compliance\nStaffing and Training\nSMB Technician Career\nClient Interaction\nContribution to Business Strategy\nGrowth & Advancement in SMBs\nJob Rotation & Multi-skilling\nNetworking in a Local Community\nResource Management\nRole Specialization\nSMB Work Culture & Dynamics\nSMB-specific Safety Protocols\nTeam Collaboration\nTraining & Development in SMBs\nUnderstanding SMB Finances\nTrane Parts\nTrane Blowers\nTrane Capacitors\nTrane Contactors\nTrane Control Boards\nTrane Fuses\nTrane Limit Switches\nTrane Miscellaneous Parts\nTrane Motors\nTrane Pilots & Ignition\nTrane Pressure Switches\nTrane Relays\nTrane Sensors\nTrane Valves\nWater & Plumbing\nBooster Pumps\nCirculator Pumps\nFaucets\nFlush Valves & Accessories\nPool Heaters\nSewage & Sump Pumps\nWater Boilers\nWater Heaters\nAbout Us\nSitemap\nTerms Of Use\nPrivacy Policy\nContact\nFacebook\nTwitter\nLinkedin\nCopyright © HVAC Tactician\n\n\n</ARTICLE_CONTENT>\n<TARGET_AUDIENCE>\nEntry-level HVAC technicians\n</TARGET_AUDIENCE>\n<USER_GUIDELINES>\nNone.\n</USER_GUIDELINES>\n<COURSE_BACKGROUND>\nCourse related information:\n- This course on temperature measurements is part of a larger learning plan on Airflow and Ductwork.\n- This will be a self-paced e-learning course.\n- The target audience for this course is entry-level technicians.\n</COURSE_BACKGROUND>\n<ideal_output>\n<summary> The article provides a comprehensive overview of different types of thermometers used in HVAC systems, including digital, analog, and wireless options. It covers their applications, features, and selection criteria, along with answers to common questions about thermometer usage in HVAC work. The content is educational in nature and includes detailed explanations of various thermometer types, their principles of operation, and practical considerations for HVAC technicians. </summary> <analysis_breakdown> 1. Content Structure and Depth: - Organized into clear sections covering different thermometer types - Includes technical explanations of operating principles - Provides practical selection criteria - Addresses common questions through FAQ section - Contains both theoretical and practical information 2. Educational Value: - Explains fundamental concepts about different measurement technologies - Provides context for when to use different types of thermometers - Includes practical considerations for tool selection - Addresses calibration and maintenance needs 3. Alignment with Course Topics: - Directly addresses digital thermometers subtopic - Covers temperature probes in detail - Includes information about infrared thermometers - Mentions calibration requirements - Missing detailed coverage of wet bulb/dry bulb measurements 4. Target Audience Appropriateness: - Written at an appropriate level for entry-level technicians - Provides clear explanations without assuming extensive prior knowledge - Includes practical applications and considerations - Balances technical detail with accessibility </analysis_breakdown> <content_evaluation> Key passages: 1. \"Different types of thermometers are used in the HVAC industry to ensure precise temperature readings\" - Establishes fundamental importance 2. \"Probe thermometers consist of a metal needle-like probe that is inserted into the substance or component being measured\" - Explains basic technology 3. \"It is recommended to calibrate your thermometer at least once a year\" - Addresses maintenance requirements Potential connections to course objectives: - Understanding different temperature measurement technologies - Selecting appropriate measurement tools - Maintaining measurement accuracy - Understanding measurement tool limitations Complexity relative to target audience: Appropriate for entry-level technicians, provides clear explanations with practical context Content type: Educational, focusing on concepts and practical application Overall relevance: Highly relevant to course objectives and appropriate for target audience </content_evaluation> <objective_brainstorm> Potential learning objectives based on content: 1. Differentiate between various types of HVAC thermometers 2. Select appropriate thermometer types for specific HVAC applications 3. Explain basic principles of temperature measurement technologies 4. Identify key factors in thermometer selection and maintenance </objective_brainstorm> <final_verdict> This article provides excellent educational content that aligns well with the course objectives and is appropriate for the target audience. The content is comprehensive, well-structured, and focuses on educational rather than product-specific information. Recommend including this content in the course materials. </final_verdict> <learning_objectives> <objective> After studying this material, students will be able to differentiate between the main types of thermometers used in HVAC systems (digital, analog, and wireless) and their specific applications. Supporting content: \"Different types of thermometers are used in the HVAC industry to ensure precise temperature readings... depending on the specific needs of an HVAC technician.\" </objective> <objective> After studying this material, students will be able to explain the key factors to consider when selecting an appropriate thermometer for specific HVAC applications. Supporting content: \"When selecting a thermometer for HVAC use, consider the following factors: Accuracy, Range, Features, Durability...\" </objective> <objective> After studying this material, students will be able to describe proper maintenance requirements for HVAC thermometers, including calibration schedules. Supporting content: \"It is recommended to calibrate your thermometer at least once a year or according to the manufacturer's guidelines. Calibration ensures that the thermometer is providing accurate temperature readings...\" </objective> </learning_objectives>\n</ideal_output>\n</example>\n</examples>\n\n

You are an experienced instructional designer tasked with analyzing content for an e-learning course and creating a refined course outline. Your goal is to summarize an article related to a search query, analyze its relevance to the course, and identify potential learning objectives if appropriate.

First, review the following information:

Article Content:
<article_content>
{article_content}
</article_content>

Course Name:
<course_name>
{course_name}
</course_name>

Target Audience:
<target_audience>
{target_audience}
</target_audience>

Course Background:
<course_background>
{course_background}
</course_background>

Rough Course Outline:
<rough_outline>
{rough_outline}
</rough_outline>

Additional User Guidelines:
<user_guidelines>
{user_guidelines}
</user_guidelines>

Search Query:
<search_query>
{search_query}
</search_query>

Your task is to analyze this content and identify potential material for the course. Follow these steps:

1. Summarize the article content:
   Provide a concise summary of the main points in the article.

2. Analyze the article:
   Evaluate the relevance and usefulness of the article content in relation to the course name, target audience, background, and rough outline. Consider how well it aligns with the course objectives and whether it's appropriate for the intended audience.

   Important: If the article is primarily focused on product features or specifications and lacks substantial educational value, clearly state this in your analysis and explain why it's not suitable for creating learning objectives.

   Address the following points in your evaluation:
   - Identify and quote key passages from the article that relate to the course.
   - List potential connections between the article and course objectives.
   - Evaluate the article's complexity relative to the target audience.
   - Determine if the content is educational or primarily product-focused.

3. Identify learning objectives (if applicable):
   If your analysis determines that the article is relevant and valuable for the course, create a list of potential learning objectives derived from the article content. These should be concepts or skills worth covering in the course that align with the overall course goals. Each learning objective should be specific, measurable, and relevant to the course.

   If the article is not suitable for creating learning objectives (e.g., it's too product-focused or irrelevant), skip this step and explain why in your analysis.

   For each learning objective, you must include the specific content from the article that supports it. Do not create learning objectives based on your own knowledge; they must be directly tied to the provided article content.

Wrap your summary, analysis and output inside the following tags:

<summary>
[Your concise summary of the article]
</summary>

<analysis_breakdown>
[Use this space to break down your analysis process, considering each aspect of the article and its relevance to the course. Show your reasoning for each step of the evaluation. It's okay for this section to be quite long.

- List key topics from the article
- Compare these topics to the course outline and objectives
- Evaluate the complexity of the content relative to the target audience
- Assess the educational value vs. product focus]
</analysis_breakdown>

<content_evaluation>
Key passages:
1. "[Quote 1]" - This relates to [course objective].
2. "[Quote 2]" - This connects to [course topic].

Potential connections to course objectives:
- [Connection 1]
- [Connection 2]

Complexity relative to target audience: [assessment]

Content type: [Educational/Product-focused]

Overall relevance: [Explanation of relevance or lack thereof]

[If not suitable for learning objectives]: This article is not suitable for creating learning objectives because [explanation].
</content_evaluation>

<objective_brainstorm>
[If applicable, your brainstormed potential learning objectives, each linked to specific content from the article]
</objective_brainstorm>

<final_verdict>
[Based on all the analysis done so far, present your final verdict on whether to include any LOs and content from this article or not.]
<final_verdict>

<learning_objectives>
<objective>
After studying this material, students will be able to [specific, measurable objective related to the course].
Supporting content: "[Exact quote or paraphrase from the article that supports this objective]"
</objective>
[Additional objectives as needed, or omit this section if the article is not suitable]
</learning_objectives>

Remember to focus on content that is directly relevant to the course and appropriate for the target audience. Be clear and explicit about cases where the article content is not suitable or relevant for the course, explaining why in your analysis. Ensure that all learning objectives are directly supported by content from the provided article.
"""


def get_relevant_info_from_article(course_name, target_audience, course_background, course_outline, search_query, article_content,
                                    user_guidelines = 'None',
                                    llm = 'gemini_2_flash'):
    """
    Get the relevant information from an article
    Args:
        user_guidelines (str): The user guidelines of the course
        search_query (str): The search query used to find the article
        article_content (str): The content of the article
        llm (str): The language model to use
    Returns:
        summary (str): The summary of the article
        analysis_breakdown (str): The analysis breakdown of the article
        content_evaluation (str): The content evaluation of the article
        objective_brainstorm (str): The objective brainstorm of the article
        final_verdict (str): The final verdict of the article
        learning_objectives (str): The list of learning objectives of the article
    """

    relevant_info_agent = Chain(llm = llm, tags = ['summary', 'analysis_breakdown', 'content_evaluation', 'objective_brainstorm', 'final_verdict', 'learning_objectives'])

    relevant_info_agent.add_message(
        role = 'user', content = extract_relevant_info_from_article_as_lo_prompt.format(
            course_name = course_name,
            target_audience = target_audience,
            course_background = course_background,
            rough_outline = course_outline,
            user_guidelines = user_guidelines,
            search_query = search_query,
            article_content = article_content
        )
    )

    response = relevant_info_agent.run()
    return response


def run_get_relevant_info_from_article(sheet, worksheet_name, course_name, target_audience, course_background, llm = 'gemini_2_flash'):
    """
    This function identifies the relevant articles and extracts relevant info as learning objectives

    :param course_name: The course name.
    :param target_audience: The target audience.
    :param course_outline: The outline of the course.
    :param course_background: The course background.
    :param llm: The language model to use.
    :return: None
    """

    preliminary_research_sheet, preliminary_research_df = get_sheet_data_and_df(sheet, worksheet_name)
    rough_outline_sheet, rough_outline_df = get_sheet_data_and_df(sheet = sheet, sheet_name = "Rough Outline")

    if 'summary' not in preliminary_research_df.columns:
        preliminary_research_df['summary'] = ''
        preliminary_research_df['analysis_breakdown'] = ''
        preliminary_research_df['content_evaluation'] = ''
        preliminary_research_df['objective_brainstorm'] = ''
        preliminary_research_df['final_verdict'] = ''
        preliminary_research_df['learning_objectives'] = ''

    # Check if this step is already done
    try:
        validate_column_values(
            df = preliminary_research_df,
            filter_column = 'article_content_0',
            validation_column = 'summary',
            require_populated = True
        )
        print('Skipping Get relevant info step. Already populated.')
        return
    except:
        pass

    # if preliminary_research_df.iloc[-1]['summary'] != '':
    #     print('Learning objectives already populated')
    #     return
    
    article_content_col_count = len([col for col in preliminary_research_df.columns if 'article_content_' in col])

    print(f"Count: {article_content_col_count}")

    # Get the course outline
    course_outline = get_outline_with_los(
        df = rough_outline_df,
        include_learning_objectives = False
    )

    # Prepare for parallel processing
    futures_map = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit tasks for each row
        for index, row in preliminary_research_df.iterrows():

            # Skip if already populated
            if row['summary'] != '':
                print(f'Skipping row {index}. Already populated')
                continue
            
            # Get the article content from the columns
            article_content = ''.join([row[f'article_content_{i}'] for i in range(article_content_col_count)])
            if not article_content:
                # print(f'Skipping row {index}. No article content found')
                continue
            
            # Submit the task
            future = executor.submit(
                get_relevant_info_from_article,
                course_name,
                target_audience,
                course_background,
                course_outline,
                row['query'],
                article_content,
                row.get('user_guidelines', 'None'),
                llm
            )

            # Map the Future to the index
            futures_map[future] = index

        # Collect the results as they complete
        total_tasks = len(futures_map)
        save_interval = 5  # how often to save (in number of completed tasks)

        if total_tasks == 0:
            raise

        # Initialize the progress tracker
        progress = SmartProgressBar(total_tasks = total_tasks, description = "Percent complete", save_interval = save_interval)

        # Now, pass only the futures (the keys) to as_completed:
        for future in tqdm(as_completed(futures_map), total=total_tasks):
            index = futures_map[future]  # retrieve the index
            response = future.result()

            # Update the df
            preliminary_research_df.loc[index, 'summary'] = response.get('summary', '')
            preliminary_research_df.loc[index, 'analysis_breakdown'] = response.get('analysis_breakdown', '')
            preliminary_research_df.loc[index, 'content_evaluation'] = response.get('content_evaluation', '')
            preliminary_research_df.loc[index, 'objective_brainstorm'] = response.get('objective_brainstorm', '')
            preliminary_research_df.loc[index, 'final_verdict'] = response.get('final_verdict', '')
            preliminary_research_df.loc[index, 'learning_objectives'] = response.get('learning_objectives', '')
            
            # Update progress
            progress.update()

            # Check if we should save
            if progress.should_save():
                print(f'Saving partial progress to sheet after {progress.completed_count} tasks completed.')
                save_to_sheet(worksheet = preliminary_research_sheet, df = preliminary_research_df)

    # Final save to sheet after all tasks
    print('All rows processed. Saving final DataFrame to sheet.')
    save_to_sheet(worksheet = preliminary_research_sheet, df = preliminary_research_df)

    return

