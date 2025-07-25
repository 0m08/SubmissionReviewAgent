from modules.chain import Chain
from tqdm import tqdm
from services.youtube_video_loader import get_transcript_with_fallback
from services.youtube_video_loader import get_yt_chapters_chunks_as_docs
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit as st
import json
from services.helper_functions import create_and_populate_columns, get_outline_with_los
from services.smart_progress_bar import SmartProgressBar
from services.sheets_service import (
    get_sheet_data_and_df,
    format_worksheet,
    create_or_read_worksheet,
    save_to_sheet,
    hide_columns_by_name,
    clear_worksheet,
    delete_worksheet,
)
from langsmith import traceable


classify_video_few_shot_examples = """<examples>
<example>
<course_info>
Course Name: Taking Temperature Measurements
Target Audience: Entry-level HVAC technicians
Course Outline:
Topic: Thermometers
  Subtopic: Digital thermometers
  Subtopic: Temperature probes
  Subtopic: Infrared thermometers & thermal imaging
  Subtopic: Gauge / meter calibration
  Subtopic: Dry bulb and wet bulb Delta T
</course_info>

<video_info>
Video Title: HVAC Training Basics for New Techs: Gauges, Pressures, Temps, Check the Charge!
Video Transcript: hey guys this is craig migliaccio from adc service tech and today what we're going over is reading compound manifold gauge sets and understanding saturated refrigerants so that you can check the refrigerant charge of an air conditioning system right now we're going to be going over a part of a powerpoint that we have for sale over at our website at ac service tech.com and here we're going to be using maybe about half of the slides that are included in the powerpoint itself so you can use this to teach technicians within your company or if you're a school you can use this within the classroom so here we go over on the left hand side you see we have a blue gauge and that's for the low pressure side of the system which is also referred to as the suction or the vapor side and you see that the pressure on the outer ring which which is the psig increments it only goes up to 350 psig over on the right hand side you see that you have a red gauge and that's for the high pressure side of the system and that's also referred to as the liquid line you can see that the pressure actually goes up to 800 so it's a much higher pressure than on our vapor gauge also on our vapor gauge we have a vacuum reading and that's in inches hg so inches of mercury a vacuum so that is in the green section right here manufacturers make compound manifold gauge sets with multiple different types of refrigerant that that can be on the gauge face itself depending on the application in this case we have r22 r4 tonight and r404a so on the gauges right here you see we're reading 217 psig on each of these gauges and as we move in through this power point we'll have these gauges measuring the pressure on refrigerant bottles and on the system itself so here we're reading the saturated temperature of r22 at any given pressure here we have the saturated temperature of r410a and so our futenai the color is pink or light rose and we have that indicated by this black line going across right here so you take your pressure you bring that into the inner ring for r4109 if you're measuring an r4 tonight system or an r410a bottle and here you see we have our 404a so you just bring your pressure to the inner ring now in this example right here we're measuring a pressure of 169 psig on both gauges so say this the the gauges are measuring the pressure on a equalized system maybe the system has a piston metering device the system's been off and so the pressure on both sides of the system is matching so once again in this case we have 169 psig if the system was r22 then you have a saturated temperature of 89 degrees so you just bring it in from the 169 psig to the green inner ring for r22 and you get 89 degrees as your saturated temperature if there was r410a in this system then you have 59 degrees and so that's shown on both gauges the the blue and the and the red high side gauge also keep in mind that these little increments are five psig increments and so right here you have a 100 and then you have five increments to get up to 125 and that's the same right here you have five to get up to 125 psig so it's just that the the low side gauge the increments are spread out whereas the red gauge are a little bit more squished because it goes up to a higher pressure also keep in mind that if you had a digital gauge set it's going to convert the pressure to saturated temperature for you so you wouldn't need to do it this way in this example you see that we're reading a pressure of 217 psig if we were to bring that into the inner ring and we had r22 in the system then the saturated temperature would be 107 degrees but if we had an r410a in the system then you can see that it would be 75 degrees so you can see it a little more clearer because the increments are more spread out on the low side gauge now let's start applying what we know so we have a low side gauge on the on the refrigerant bottle and this refrigerant bottle is an r410a bottle because it's pink or light rose and in this case you can see in the inside of the bottle we have liquid in the bottle and vapor so this bottle is within a saturated state where liquid and vapor both exist at the same time the pressure being applied on the gauge by the refrigerant is 217 psig and you can see a little bit clearer with our magnifying glass right here you bring it into the inner ring and you read 75 degrees as a saturated temperature of the refrigerant now we're also measuring on the outside of the bottle with a temperature meter and the temperature is measuring 75 degrees so if this bottle was within a room that was 75 degrees for several hours the room and the bottle and the refrigerant within the bottle would all be at 75 degrees so that is how you could check to see if you have pure refrigerant within the bottle and if the refrigerant is in the saturated state within the bottle everything should match in this case we'll take it a little step further and you see that this bottle is fairly full you have vapor and liquid in the bottle whereas this right here you have mainly vapor and a small amount of liquid it's nearly empty however the refrigerant in both bottles is still saturated and if the refrigerant is saturated within the bottle you're going to be applying the same pressure to the gauge regardless of how much refrigerants inside of here so as long as you have some liquid in the bottle and the bottle is not completely empty with only vapor then you know that you're going to have the same matching pressure so in this case it's 217 psig bring it into the r4 tonight inner ring for the saturated temperature and we measure 75 degrees that matches the surrounding air temp of 75 degrees and the temperature that the bottle is at in this picture the bottle on the left contains one ounce of liquid refrigerant and the bottle on the right contains no liquid refrigerant so it's only vapor the surrounding air temperature is 75 degrees as indicated on both temperature meters so the saturated temperature of the bottle on the left matches the surrounding air temperature which is 75 degrees however on the bottle on the right the pressure converted to saturated temperature measures only 40 degrees as a saturated temperature even though our temp meter is measuring 75 degrees this shows that the refrigerant bottle is just about empty and no longer contains saturated refrigerant but only vapor so in this example we have pt charts and remember that the saturated temperature on the gauge face is just a simplified pt chart that's overlaid onto that gauge face so to read an actual pt chart you need either temperature or pressure to start and over on the left hand side of the pt chart you're going to have either pressure or temperature in this case we're using temperature on the left hand side you see that the this temperature starts at zero and continues down to get to 68 degrees and then it just restarts over here so this is just a continuation of the same pt chart to work with this pt chart we just need a temperature so if we have a surrounding air temperature of 78 degrees then we know that the bottle should be at 229 psig if there's our fort tonight in that bottle in the case of r22 if we measured a green r22 bottle and it was 78 degrees surrounding the bottle then the pressure within the bottle should be 139 degrees if it actually is r22 in that tank so we can use what we learned already to determine what refrigerant is in a used recovery bottle and if it happens to be contaminated reusable recovery bottles are identified by the colors gray and yellow so yellow is on the neck and then gray is in the body of the tank you should always make sure to note what refrigerants inside of a recovery bottle so you don't get confused and in this case you see that we have a note with r410a if there happens to not be a tag on that refrigerant bottle the refrigerant in the bottle needs to be identified this can be done by comparing the temperature surrounding the bottle so that's right here 78 degrees so you can compare that with the saturated temperature of the refrigerant inside the bottle so right here so in this case you see that it matches so we're reading a saturated temperature of r for tonight at 78 degrees and that matches 78 degrees right here if we were to do that on a pt chart we can just come over and if we measure 229 psig we bring it over to the temperature column of 78 degrees and you see that that matches even if the bottle has a refrigerant tag make sure to check the saturated temperature of the refrigerator in the bottle before use there could be a problem with that tank where you have air that may have got sucked into that tank maybe from a leak in a system or something like that and that may have contaminated the bottle but now you're getting ready to reuse it again you want to check the pressure and the saturated temperature of that recovery bottle and compare that to the temperature surrounding the bottle just to make sure that the refrigerator is not contaminated before you use that bottle again here's an example of a contaminated refrigerant bottle where we have 259 psig and that's right here and the pressure converted to our futenai for the saturated temperature is 86 degrees so it's a little hard to make that out but that is 86 degrees as a saturated temperature however the temperature surrounding the bottle is 78 so if you have air contaminating the bottle the pressure will be higher and so once again here's another example using the pt chart 259 and you convert that to the saturated temperature of r for tonight it's 86 however the bottle temperature is actually 78 even though the bottle has remained at a steady temperature say in a room for several hours now we're going to apply what we know to measuring the pressures on an actual hvcr system so it's possible to determine the type of refrigerant in an often equalized system so you see that our pressure on both the the high side gauge and the loci gauge matches and as long as you have a piston orifice which is the small little fixed orifice at usually at the beginning of the evaporative coil if you have one of those and the system's been off then these should be equalized if you have a thermostatic expansion valve it may not completely be equalized just remember that if the high side and the low side gauge match in pressure the entire system is within the saturated state right now all the refrigerant is saturated liquid and vapor exists throughout the whole system the pressure right here on the gauge measures 222 psig and if you convert that to the saturated temperature you measure 76 degrees and that matches the surrounding air temperature 76 degrees and the temperature applied on the actual line itself and in this case it's like the perfect scenario but uh you know normally it's not that perfect but in this case the inside temperature is also 76 degrees but this particular scenario is works great on a package unit where the refrigerant is entirely outside in that outdoor unit on a split system it can be a little harder because you have an indoor temperature that differs from the outdoor temperature the type of refrigerant can be determined using the the rating plate so it'll say the type of refrigerant on the outdoor rating plate you can also look if there's a tag on the compressor on the outdoor unit and you can also look on the head of the txv if there is one equipped so it's a thermostatic expansion valve if there's equipped it should say the refrigerant in the system in this case you see that the outdoor unit tag is worn off that could just be due to the sunlight or something like that but it's just completely bleached white and it could just be gone that that's another thing that could happen in that case you need to identify the type of refrigerant that you're working with before you just go ahead and slap the gauges on and start trying to measure your your refrigerant charge you can also use a refrigerant analyzer and that would tell you what refrigerants in the system but in this case we're just going by the gauges and temperature in this case we're giving an example of a contaminated refrigerant charge within the system so the refrigerant is contaminated with air in this scenario and you see that we're reading a pressure that's very close to 300 psig it's right around say 297 psig if we were to bring that into the inner ring you see that it's much higher than the temperature surrounding the unit so in this case we're reading a saturated temperature of 95 degrees for r410a and we're reading a air temperature and line temperature of 76 degrees so you can tell that this the refrigerant in this system is definitely contaminated and it's usually contaminated with air if it's going to be that high or nitrogen something like that here's an example of a system that's severely low in refrigerant and in fact so much refrigerant has leaked out that there's no more liquid refrigerant in the system so it's no longer in the saturated state the system is off and we see that we read a pressure of 110 psig we bring that into the saturated temperature of r4 tonight because that's what's stated on the reading plate and you see a saturated temperature of 36 degrees that is way way lower than 76 degrees and that's what we should have 76 degrees you see we're measuring the temp here and outside and so that is a indication that this entire system is severely low due to a refrigerant leak without even having to turn the system on we've skipped ahead in the powerpoint and now we're checking the charge on a running system however before you do that you want to make sure that you have good indoor airflow matched to the system size on the outdoor unit rating plate you also want to make sure that you know what metering device is in the system so you know how to check the charge you also want to know what refrigerants in the system and you want to also allow the system to run for say 10 15 minutes before checking the actual charge however you do monitor the gauges so i go over all that stuff in the powerpoint but in this case you see that we have a thermostatic expansion valve as the metering device at the indoor unit we're going to be using the sub coin method so now we're applying what we know and so you know how to read the gauge sets and you know how to convert that to saturated temperature so the red high side gauge is actually measuring the saturated temperature of the refrigerant in the outdoor unit while the system is running the blue gauge is measuring the saturated temperature of the refrigerant in the indoor coil while the system is running but in this case we're going to be using the sub coin method because we have our thermostatic expansion valve we need to check the pressure on the high side gauge we're going to convert that to saturated temperature and we also need to take a measurement on the small liquid line and you see that we're measuring 92 degrees right there so our saturated temperature for r410a refrigerant on this particular unit while it's running is 104 degrees minus the liquid line temperature is 92 degrees and we're left with a sub coin of 12 degrees so this is how we check the charge of a system with the thermostatic expansion valve is with the sub cooling method the target sub coin can be found on the outdoor unit rating plate so in this case it says a txv sub coin of 12 degrees in some cases it may have multiple different txv sub coins depending on the outdoor temperature you may also find the the target sub coin on the inside of this shroud right here it may not be on the rating plate might be on the inside of the shroud if there's no target sub coin posted for the for the unit in air conditioning mode a target sub coin of 11 degrees can be used on single speed or two speed units so now we need to compare the actual sub coin to the target sub coin so you find your sub coin from the saturated temperature on the red high side gauge minus the actual liquid line temperature so that's the small liquid line if the actual sub cooling is less than the target sub cooling the need to add refrigerant to the system if the actual sub cooling is higher than the target sub coin then you need to recover refrigerant and the correct refrigerant level is if you have the actual sub cooling is within plus or minus three degrees of the target sub coil so preferably it's right on the target sub coin or just slightly higher maybe one degree higher than the target sub coin so that would indicate a correct refrigerant charge level in the power point we have detailed examples of a system that's low in refrigerant a system that's overcharged and one that's a correct charge but now let's go move on to the total superheat method so if we have a system that has a fixed orifice and a fixed orifice could be a piston or a capillary tube metering device then we're going to use the superheat method and since we're measuring it at the outdoor unit it's going to be considered the total superheat method so in this case we have our pressure converted to saturated temperature and we have a saturated temperature of 42 degrees we're also going to take our actual line temperature on the vapor line and you see that we're measuring a temperature of 56 degrees in this case it's reverse from sub cooling so you got to remember that it's the actual line temperature minus the saturated temperature when we were working in sub cooling it was the saturated temperature minus the actual temperature so you just got to remember that they are reversed from each other so we take 56 degrees as our actual vapor line temperature minus the saturated temperature on the blue low side gauge which is 42 degrees and we come up with 14 degrees of total super heat now we need to compare the total super heat to the target superheat target superheat is found by first measuring the indoor wet bulb temperature so at the indoor unit and also the dry bulb temperature surrounding the outdoor unit so it's actually the air that's getting sucked into the outdoor unit and getting blown at the top so you got to take it down low so you don't want it to get affected by the the heat that's being rejected above the unit so in this case we're reading an 85 degree dry bulb standard air temperature entering into the outdoor unit you can input the wb and the db temperature into a super heat chart or you can enter into an app a digital calculator or a calculation tool like we have over the website at ac serverstick.com in order to measure the indoor wet bulb temperature you're going to have to measure within just a few feet of the evaporator coil so in the return duct if you have a a grill that's really far away from where the actual the unit is the the temperature can change the wet bulb temperature can change by the time it gets to here maybe that return duct is running through an attic or something like that so you really want to get it as close to the air handler as possible to measure the db temp you want to make sure to measure about a foot away from the outdoor coil in the shade away from the hot discharge air exiting the coil and this can be done with a standard temperature reader make sure to measure the indoor weapon temperature with a digital psychrometer you could also do it with the sling cyclometer or also a standard temp reader with a wet sock over the bead temp sensor but we have other videos on that link down in the description section below so you can go ahead and check them out in this example you have a db temp so that's a dry bulb outdoor ambient temp so it's either oa it could be referred to or db that's the outdoor ambient temperature 85 degrees our indoor wet bulb temperature is 68 degrees we input them onto a target superheat chart and we have our indoor wet bulb temperature is 68 degrees so you bring that down and then you have your outdoor db temp and you bring that across and you come up with 19 degrees as your target superheat depending on what you use to calculate your target superheat it may be say one degree off from each other you know you could use a digital manifold gauge set and that will come up with the target super heat automatically for you you could use a calculator an app you know a calculation charts but we have a full article on all that stuff over at the website so you can read that over ac servicestick.com in this case we have a target superheat of 19 degrees and that's presently you got to remember that as a system runs your indoor wet bulb temperature is going to lower because it's taking the humidity out of the building so your target superheat is going to change while the system is running so you constantly have to monitor your indoor wet bulb temperature while you're checking the refrigerant charge and while that system is running now we need to compare our total superheat to the target superheat and you see that we have the actual line temperature on the vapor line minus the saturated temperature found on the blue low side gauge and that's how we come up with our total superheat if the unit was low in refrigerant we needed to add some refrigerant you know obviously you're going to find your leak first and see if you can fix that leak but if you need to add refrigerant your total superheat is going to be higher than your target superheat if you are over charged and you need to recover some refrigerant your total superheat is going to be less than your target superheat and if you have the correct refrigerant level your total superheat is going to be within plus or minus two degrees of your target superheat and once again we have detailed examples in our powerpoint discussing each one of these so we have our undercharge our overcharged and our correct charge if you want to learn more about preparing a system for refrigerant checking the refrigerant charge and also troubleshooting check out our book the refrigerant charging and service procedures for air conditioning so we also have a thousand question workbook that helps you to retain your knowledge that you're learning from the book now this workbook also has an answer key so you can check your answers so it's a self-study guide and we also have our quick reference cards that can be used out in the field right next to the unit while you're checking the charge or troubleshooting we have all these resources available at our website at ac servicestick.com ac book we also have all of our physical products available over amazon and also ebay and we have our ebook over at the apple bookstore and also on google play hope you enjoyed yourself and we'll see you next time at aec service tech channel
</video_info>

<analysis>
Upon comparing the video content to the provided course outline, several key observations emerge:

1. **Course Focus vs. Video Focus**:
   The course, "Taking Temperature Measurements," targets entry-level HVAC technicians, focusing on understanding various thermometers, probes, infrared thermometers, thermal imaging, and gauge/meter calibration. It aims to teach fundamental temperature measurement techniques and how to use these devices to accurately gauge conditions such as dry bulb and wet bulb temperatures.

   In contrast, the video primarily focuses on understanding and interpreting manifold gauge sets, pressure-temperature relationships, identifying refrigerant charge conditions, and using pressure measurements to infer refrigerant saturation and contamination. While it occasionally touches on measuring line temperatures with a temperature meter, the core emphasis is on pressure readings, PT charts, and the concept of saturation rather than on the direct use and selection of temperature measurement instruments as outlined in the course.

2. **Relevance to Course Topics**:
   - **Digital Thermometers & Temperature Probes**: The video only briefly mentions taking temperature readings (e.g., measuring the temperature of a refrigerant bottle or a refrigerant line), but does not provide comprehensive guidance on using digital thermometers or temperature probes in a systematic instructional manner.

   - **Infrared Thermometers & Thermal Imaging**: The video does not address infrared thermometers or thermal imaging devices at all.

   - **Gauge / Meter Calibration**: While the video uses manifold gauges extensively, it focuses on interpreting their readings rather than instructing viewers on calibrating these instruments to ensure accurate temperature measurements.

   - **Dry Bulb and Wet Bulb Delta T**: Although wet bulb and dry bulb concepts are indirectly referenced when determining target superheat, the discussion is more about refrigerant charge verification rather than in-depth measurement techniques for air properties. The video does not explicitly instruct viewers on how to measure dry/wet bulb temperatures using thermometers or psychrometers beyond a surface-level mention.

3. **Appropriateness for the Target Audience**:
   Entry-level technicians, as envisioned by the course, should first learn how to handle basic temperature-measuring devices. The video’s heavy focus on interpreting pressure readings and saturated states, while useful in HVAC troubleshooting, is more specialized and advanced. It does not offer a clear, step-by-step approach to using common HVAC thermometers or IR tools—skills that are central to the course outline.

4. **Potential to Enhance the Course**:
   While the video could be useful as supplemental material for technicians who have already grasped the basics of using thermometers (to show them how temperature measurements tie into overall system diagnostics), it does not cleanly integrate with the primary learning objectives of the course. It might provide context for why accurate temperature measurements matter when diagnosing system charge, but the content does not directly enhance the foundational topics listed.

In summary, the video’s main educational value lies in advanced system analysis using manifold gauges and PT charts, rather than directly teaching how to take temperature measurements with various thermometers or how to calibrate those instruments as laid out in the course outline. It could be considered supplemental but is not strongly aligned with the core course topics.

While the video offers valuable insights into HVAC system analysis and refrigerant charge checking, it does not significantly align with the core course topics centered on temperature measurement devices and techniques. As such, it is not well-suited for shaping the course outline.
</analysis>

<verdict>
IRRELEVANT
</verdict>

</example>

<example>

<course_info>
Course Name: Comfort
Target Audience: Entry-level HVAC technicians
Course Outline:
Topic: Temperature & Heat Transfer
  Subtopic: Role of temperature in comfort
  Subtopic: BTU - Definition and use
  Subtopic: Latent heat
  Subtopic: Sensible heat
  Subtopic: Temperature
  Subtopic: Regional temperature considerations and comfort

Topic: Humidity
  Subtopic: Role of humidity in comfort
  Subtopic: Fundamentals of humidity
  Subtopic: Adjusting system performance for humidity control

Topic: Air Movement
  Subtopic: Basic energy rules
  Subtopic: Mass flow vs. Volume flow
  Subtopic: System & distributed airflow
  Subtopic: Infiltration & exfiltration
  Subtopic: Design ventilation (positive, negative, & balanced)

Topic: Indoor Air Quality
  Subtopic: Ventilation - occupant health
  Subtopic: Contaminants
  Subtopic: Odor control
</course_info>

<video_info>
Video Title: Heat and Comfort Basics - 3D
Video Transcript: in this video we're going to talk about heat transfer in a building specifically heat transfer in and out of a residential home these are known as heat losses when heat is leaving the structure and heat gains when heat is entering the structure here we show a basic gas furnace setup in a garage as well as the condensing unit outside when the gas furnace is operating in heating mode it adds BTUs of heat to the home in order to balance out the BTUs that are leaving when it's in air conditioning mode the opposite happens it removes BTUs of heat from the home to balance out the number of BTUs that are added this begs the question though what is a BTU BTU stands for British thermal unit which is a simple measure of heat and it is equal to the amount of heat transfer required to change the temperature of one pound of water by one degree Fahrenheit this applies to both Heating and Cooling and we use it as an everyday measure to discuss the movement of heat by quantity now let's talk about some of the specifics of how heat is transferred into and out of a home the first and simplest form of heat transfer is called conduction this occurs when molecules come into direct contact with one another transferring their heat from the higher temperature matter to the lower temperature matter heat will continue to transfer until the temperature of the materials reaches equilibrium meaning that the same temperature we use insulation to oppose conduction reducing the rate of heat transfer insulation that opposes conduction is rated by R value R represents the resistance to heat movement the higher the R value the lower the heat transfer rate in this example we show a warm house and a cold attic heat is going to transfer through the ceiling and insulation from the hotter to the colder or the higher temperature to the lower temperature the opposite direction of heat flow also occurs here we show a hotter attic and a 75 degree Fahrenheit indoor temperature it's very common here in Florida where we have hot addicts with cool indoor temperatures insulation helps reduce this rate of transfer the heat transfer will increase or decrease proportionally based on the temperature differential a higher differential means there will be a higher heat transfer rate and a lower differential means there will be a lower heat transfer rate additional insulation will decrease the heat transfer rate and less insulation will increase the heat transfer rate all of these transfers again work towards equilibrium the only time heat transfer stops is when the temperatures are equal here we show some very cold outdoor temperatures to show how heat could be transferred via conduction through the wall into a snow bank again moving from higher temperature to lower temperature convection is another type of heat transfer convection occurs when molecules themselves move and bring their heat with them this is true in fluids in buildings this would primarily be heat carried through the air although convection can also occur in liquids in the case of ponds and streams or think of when you're filling a bathtub with hot water and you kind of mix the hot water around in order to balance out the temperature that mixing is convection inside a home we often see undesirable convection in simple cases like when doors and windows are open if the temperature outside is lower than the temperature inside and we were to open a door we would lose heat from the home as those molecules move outside or if the wind were to blow through the door with a cool breeze that would result in convection carrying lower temperature molecules into the home reducing the home's average temperature anytime air is leaving or entering a home we're either gaining or losing heat via convection we call this infiltration when air comes in or exfiltration when air leaves some other common areas in which we can lose or gain heat via convection are around things like lights or attic vents the third type of heat transfer is radiation which can be more difficult for us to understand because radiation happens via electromagnetic waves some of the most common ways that we observe radiation are by feeling the warmth from the Sun or being in front of a fireplace whenever we have Windows in a home the inside of the home is heated by electromagnetic radiation from the sun when sunlight enters and beats on a Surface as shown here radiant heat can transfer both directions though that's why when you stand in front of a cold wall you'll feel yourself being cooled even if the air temperature in the space is comfortable that's because radiant heat transfer occurs between bodies of higher temperatures and lower temperatures depending on the distance in this case it happens when our warmer bodies transfer heat to a cold wall but most commonly it happens the other direction when there's a hot surface or a flame or the sun transferring to our bodies via radiation radiation always happens line of sight now let's talk about sensible and latent heat sensible heat is heat that we can measure with a thermometer and latent heat is heat that we cannot measure with a thermometer here we show a common example of latent heat transfer we know that when we're boiling a pot of water much more energy is required to change the water from liquid water to steam than is required to increase the temperature of the water tooth boiling point it takes one BTU to raise the temperature of a pound of water by one degree Fahrenheit but it takes about 970 BTUs to change that same pound of water from liquid to a vapor the same is true when melting or making ice there's much more energy contained in the actual conversion of the water to ice or steam than by merely raising the temperature the amount of heat it takes to change a solid to liquid and vice versa is called the latent heat of fusion for water that's about 144 BTUs per pound it's latent heat because its hidden heat we're adding or removing BTUs but we're not seeing a change in temperature the amount of heat it takes to change from a liquid to a vapor and vice versa is called the latent heat of vaporization here we show two pots of boiling water one with a larger flame and one with a smaller Flame the larger flame adds more heat to the water than the smaller flame and will apply more BTUs of heat over a shorter period of time so the phase change will happen more quickly in the pot with a larger flame than the smaller flame notice how their temperature stays the same despite the difference in the added heat content this illustrates latent heat that energy is going to the change of phase not to a change in temperature so when we cool a home in a typical climate in the U.S we're also removing moisture from the home air that enters the home contains water vapor and latent heat which affects the relative humidity of the space and can negatively affect human comfort here we show air which contains water vapor entering a home via convection you will notice that the cooler air sinks and the warmer air rises many people will say that heat rises but they're actually noticing that hotter air is less dense than colder air so it floats naturally in colder air colder air is heavier and sinks in hotter air heat itself doesn't rise or fall but matter that is heated or cooled can float or sink in the same matter you can see the air moving into the return of the HVAC system which will take it into the unit here when air passes over the evaporator coil during the cooling process water condenses on the coil water vapor becomes a liquid this type of transfer is called latent heat transfer because we can't measure the energy utilized with the thermometer like we mentioned before much of that energy goes towards the phase change while the coil remains at the same temperature similar to how the pot of water stays at the same temperature during the boiling process in order to remove humidity from a home it's necessary to have the evaporator coil drop to a temperature below the dew point of the air passing over it in very simple terms the colder the evaporator coil is the more moisture it will remove from the home for an air conditioner to be ideal at removing latent heat or moisture from the home the evaporator coil needs to be at a low enough temperature and the air conditioner needs to run humidity is only removed from a home when the air conditioning system is running to calculate how much heat enters or leaves a home air conditioning contractors use a manual which is now commonly integrated in the software programs called manual J published by the air conditioning contractors of America this manual helps us use many local conditions and many other design factors in the home to design for how many BTUs the system must add or remove from the space in order for humans to remain comfortable just as BTUs can enter and leave the home via conduction convection and radiation between indoors and Outdoors human occupants also add heat to the structure human occupants radiate heat to colder surfaces conduct heat into objects and air molecules they touch generate convection when they move around and add latent heat when they exhale these are internal gains that need to be considered when doing load calculations and sizing equipment this video is a very simple overview of how heat moves in and out of a home and some of the basic principles in future videos in this series we'll be covering the residential HVAC design process and much more thanks for watching our video If you enjoyed it and got something out of it if you wouldn't mind hitting the thumbs up button to like the video subscribe to the channel and click the notifications Bell to be notified when new videos come out HVAC school is far more than a YouTube channel you can find out more by going to hvacreschool.com which is our website and hub for all of our content including Tech tips videos podcasts and so much more you can also subscribe to the podcast on any podcast app of your choosing you can also join our Facebook group if you want to weigh in on the conversation yourself thanks again for watching [Music] thank you [Music]

</video_info>

<analysis>
The given course, “Comfort,” is designed for entry-level HVAC technicians and covers topics around temperature, humidity, air movement, and indoor air quality with a strong emphasis on understanding how these factors influence human comfort. The subtopics include temperature and heat transfer fundamentals, the role of BTUs, latent and sensible heat, humidity control, and basic principles of air movement and ventilation.

When we compare the course outline to the video content, the alignment is strong:

**Alignment with Course Topics:**
- **Temperature & Heat Transfer:**
  The video provides a comprehensive explanation of how heat moves into and out of a building. It covers the three modes of heat transfer (conduction, convection, and radiation) and directly addresses how these processes affect building temperature. It also explains BTUs and the principle of heat flow from warmer to cooler areas, all of which align perfectly with the course outline’s focus on understanding the role of temperature in comfort and BTUs.

- **Latent and Sensible Heat:**
  The video clearly differentiates between latent and sensible heat, using understandable examples such as phase changes in water (boiling, condensation) and their impact on comfort and HVAC operation. This matches the course’s need to cover latent and sensible heat as subtopics crucial to understanding comfort levels.

- **Humidity and its Impact on Comfort:**
  The video discusses the removal of moisture (latent heat) by the evaporator coil, showing how humidity is controlled and how it influences comfort. Though it doesn’t delve deeply into system adjustments for humidity control, it sets a strong foundational understanding that can be expanded upon in the course.

- **Air Movement and Infiltration/Exfiltration:**
  While the video does not deeply explore mass flow versus volume flow, it does discuss infiltration, exfiltration, and how air movement can carry heat into or out of a home. This serves as a good introduction to understanding how airflow influences comfort through heat transfer and the role of proper ventilation and infiltration control.

- **Indoor Air Quality (IAQ):**
  The video does not explicitly cover contaminants or odor control, but it establishes the basic idea that occupants contribute to internal gains and affect indoor conditions. Although this connection to IAQ is weaker, the foundational knowledge of how air movement and temperature differences can bring in outside air (along with potential contaminants and moisture) is still relevant background information.

**Appropriateness for the Target Audience:**
- **Entry-level Technicians:**
  The video uses clear, basic explanations and visual examples suitable for beginners. It does not assume advanced technical knowledge and focuses on fundamental principles. This approach matches the target audience’s level.

**Potential to Enhance or Expand the Current Course Outline:**
- The video provides excellent real-world context for heat transfer principles, the concept of BTUs, and how these fundamentals tie into comfort.
- Its introduction to latent and sensible heat transfer, and the explanation of humidity’s role in comfort, directly reinforces key course subtopics.
- While some course topics like IAQ contaminants are not covered, the fundamentals provided in the video create a strong foundation upon which these more specific topics can be built.

**Quality and Depth of Information:**
- The video is thorough yet accessible, using everyday analogies (like boiling water) to explain complex concepts.
- It covers a wide range of fundamental concepts that are directly relevant to achieving comfort—temperature differentials, heat transfer modes, role of insulation, BTUs, latent vs. sensible heat, and the fundamentals of moisture removal.

Overall, the video strongly supports core portions of the course outline, especially the “Temperature & Heat Transfer” and “Humidity” topics. It provides conceptual clarity and practical understanding, making it a strong candidate as either required viewing or supplemental material within the course.

The video aligns well with the course topics on temperature, BTUs, latent and sensible heat, and humidity’s role in comfort, providing clear and foundational explanations suitable for entry-level HVAC technicians.

</analysis>

<verdict>
RELEVANT
</verdict>

</example>

</examples>

THE ABOVE WERE EXAMPLES OF THE TASK AT HAND

YOUR ACTUAL TASK BEGINS BELOW:
"""


classify_video_prompt = """You are tasked with analyzing a video transcript to determine its relevance for a course outline. You will be provided with course information and video details. Your goal is to assess whether the full video or parts of it can be used to shape the course outline.

First, review the course information:

<course_info>
Course Name: {course_name}
Target Audience: {target_audience}
Course Outline:
{course_outline}
</course_info>

Now, examine the video details:

<video_info>
Video Title: {video_title}
Video Transcript: {video_transcript}
</video_info>

To analyze the video transcript:
1. Carefully read through the entire transcript.
2. Identify key topics, concepts, or information presented in the video.
3. Compare these elements to the course outline and target audience.
4. Consider how well the video content aligns with the course objectives and depth of coverage needed for the target audience.
5. Determine if the full video or specific parts could contribute to shaping the course outline.

Provide your reasoning on why the video is or is not relevant for the course. Consider the following aspects:
- Alignment with course topics
- Appropriateness for the target audience
- Potential to enhance or expand upon the current course outline
- Quality and depth of information presented

After your analysis, present your final verdict on whether the video is relevant or irrelevant for shaping the course outline.

Format your response as follows:

<analysis>
[Your detailed analysis and reasoning here]
</analysis>

<verdict>
[State whether the video is RELEVANT or IRRELEVANT]
</verdict>

Do not enter a verdict such as "PARTIALLY RELEVANT", "CONDITIONALLY RELEVANT", etc. Only two valid options for verdict are "RELEVANT" and "IRRELEVANT"
Ensure your analysis is thorough and your verdict is clear and well-justified based on the provided information.
"""


@traceable(metadata={
    "agent_name": "course_outline",
    "step_name": "Retrieve Video Transcripts",
    "function_name": "run_get_transcripts",
    "user_id": st.session_state.get("role", "anonymous")
})
def run_get_transcripts(sheet, worksheet = "Videos Research"):
    """
    This function will get the video transcripts for all the rows and add them to sheet.

    :param sheet: The sheet object.
    :param worksheet_name: The worksheet name.
    :return: None
    """

    # Read or create dataframes
    videos_research_sheet, videos_research_df = get_sheet_data_and_df(sheet, worksheet)

    if "video_transcript_0" not in videos_research_df.columns:
        videos_research_df["video_transcript_0"] = ""

    if videos_research_df.iloc[-1]["video_transcript_0"] != "":
        print("Transcripts already populated.")
        return

    total_tasks = videos_research_df.shape[0]
    save_interval = 5

    # Initialize the progress tracker
    progress = SmartProgressBar(total_tasks = total_tasks, description = "Percent complete", save_interval = save_interval)

    for index, row in tqdm(videos_research_df.iterrows(), total = total_tasks):
        
        # Skip if already present
        if row['video_transcript_0'] != '':
            print(f'Skipping row {index}. Transcript already populated')
            progress.update()
            continue

        try:
            transcript = get_transcript_with_fallback(
                video_id = row['video_id'],
                return_text_only = False
            )

            transcript_json_str = json.dumps(transcript)
        except Exception as e:
            transcript_json_str = "" #str(e)

        # Save to df
        videos_research_df = create_and_populate_columns(
            df = videos_research_df,
            text = transcript_json_str,
            specific_index = index,
            col_base_name = "video_transcript"
        )

        # Update progress
        progress.update()

        # Check if we should save
        if progress.should_save():
            print(f'Saving partial progress to sheet after {progress.completed_count} tasks completed.')
            save_to_sheet(worksheet = videos_research_sheet, df = videos_research_df)

    save_to_sheet(worksheet = videos_research_sheet, df = videos_research_df)

    column_names = [column_name for column_name in videos_research_df.columns if "video_transcript" in column_name]

    # Hide the columns
    hide_columns_by_name(worksheet = videos_research_sheet, column_names = column_names, df = videos_research_df)

    return


@traceable(metadata={
    "agent_name": "course_outline",
    "step_name": "Check Video Relevance",
    "function_name": "classify_video",
    "user_id": st.session_state.get("role", "anonymous")
})
def classify_video(course_name, target_audience, course_outline, video_title, video_transcript, llm = "gemini_2_flash"):
    """
    Processes and classifies a video based on its relevance to a course outline.

    :param course_name: The name of the course.
    :param target_audience: The target audience of the course.
    :param course_outline: The outline of the course.
    :param video_title: The title of the video.
    :param video_transcript: The transcript of the video.
    :param llm: The language model to use.
    :return: The agent's analysis and verdict.
    """

    classify_video_agent = Chain(llm=llm, tags=['analysis', 'verdict'])
    classify_video_agent.add_message(
        role="user",
        content=(
            classify_video_few_shot_examples 
            + classify_video_prompt.format(
                course_name=course_name,
                target_audience=target_audience,
                course_outline=course_outline,
                video_title=video_title,
                video_transcript=video_transcript
            )
        )
    )
    response = classify_video_agent.run()

    return response



def run_classify_video(sheet, worksheet_name, course_name, target_audience, llm = "gemini_2_flash"):
    """
    This function classifies all the videos in the sheet

    :param sheet: The sheet object.
    :param worksheet_name: The worksheet name.
    :param course_name: The name of the course.
    :param target_audience: The target audience of the course.
    :param llm: The language model to use.
    :return: None
    """

    # Read the sheet and df
    videos_research_sheet, videos_research_df = get_sheet_data_and_df(sheet = sheet, sheet_name = worksheet_name)
    rough_outline_sheet, rough_outline_df = get_sheet_data_and_df(sheet = sheet, sheet_name = "Base Outline")

    if 'video_analysis' not in videos_research_df.columns:
        videos_research_df['video_analysis'] = ''
        videos_research_df['video_verdict'] = ''

    # Check if this step is already done by checking last row is populated
    if videos_research_df.iloc[-1]['video_verdict'] != '':
        print('Videos already classified')
        return

    # Get the course outline
    course_outline = get_outline_with_los(
        df = rough_outline_df,
        include_learning_objectives = True
    )

    # Get video_transcript column count
    video_transcript_col_count = len([col for col in videos_research_df.columns if 'video_transcript_' in col])

    # Prepare for parallel processing
    futures_map = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit tasks for each row
        for index, row in videos_research_df.iterrows():

            # Skip if already analyzed
            if row['video_analysis'] != '':
                print(f"Skip. Already analyzed video {index}: {row['title']}")
                continue

            if row['video_transcript_0'] == "":
                print(f"Skipping row {index}. No Transcript found in video_transcript_0 column.")
                continue
            
            # Get the transcript
            timestamped_transcript = json.loads(
                ''.join(
                    [row[f'video_transcript_{i}'] for i in range(video_transcript_col_count)]
                )
            )
            video_transcript = ' '.join([str(item['text']) for item in timestamped_transcript])

            
            # Submit the task
            future = executor.submit(
                classify_video,
                course_name,
                target_audience,
                course_outline,
                row['title'],
                video_transcript,
                llm = llm
            )

            # Map the Future to the index
            futures_map[future] = index

        # Collect the results as they complete
        total_tasks = len(futures_map)
        save_interval = 5  # how often to save (in number of completed tasks)

        # Initialize the progress tracker
        progress = SmartProgressBar(total_tasks = total_tasks, description = "Percent complete", save_interval = save_interval)

        # Now, pass only the futures (the keys) to as_completed:
        for future in tqdm(as_completed(futures_map), total=total_tasks):
            index = futures_map[future]  # retrieve the index
            response = future.result()

            # Update the df with analysis and verdict
            videos_research_df.loc[index, 'video_analysis'] = response['analysis']
            videos_research_df.loc[index, 'video_verdict'] = response['verdict']

            # Update progress
            progress.update()

            # Check if we should save
            if progress.should_save():
                print(f'Saving partial progress to sheet after {progress.completed_count} tasks completed.')
                save_to_sheet(worksheet = videos_research_sheet, df = videos_research_df)

    # Final save to sheet after all tasks
    print('All rows processed. Saving final DataFrame to sheet.')
    save_to_sheet(worksheet = videos_research_sheet, df = videos_research_df)

    # Hide analysis and verdict columns
    columns_to_hide = ['video_analysis', 'video_verdict']
    hide_columns_by_name(worksheet=videos_research_sheet, column_names=columns_to_hide, df=videos_research_df)

    return



def run_chunk_videos(sheet, videos_research_worksheet_name, video_chunks_worksheet_name, llm = "gemini_2_flash"):
    """
    This function chunks all the videos marked as relevant

    :param sheet: The sheet object.
    :param videos_research_worksheet_name: The worksheet name of the videos research worksheet.
    :param video_chunks_worksheet_name: The worksheet name of the video chunks worksheet.
    :param llm: The language model to use.
    :return: None
    """

    # Read or create dataframes
    videos_research_sheet, videos_research_df = get_sheet_data_and_df(sheet, videos_research_worksheet_name)
    video_chunks_sheet, video_chunks_df = create_or_read_worksheet(sheet, video_chunks_worksheet_name, rows = 1000, cols = 20)

    if 'video_id' not in video_chunks_df.columns:
        video_chunks_df['video_id'] = ''
        video_chunks_df['video_title'] = ''
        video_chunks_df['chapter_title'] = ''
        video_chunks_df['metadata'] = ''
        video_chunks_df['text_0'] = ''

    # Correct mapping for channel_title
    video_id_to_channel_title = dict(zip(videos_research_df['video_id'], videos_research_df['channel_title']))

    # Get video_transcript column count
    video_transcript_col_count = len([col for col in videos_research_df.columns if 'video_transcript_' in col])

    # Prepare for parallel processing
    futures_map = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit tasks for each row
        for index, row in videos_research_df.iterrows():

            video_id = row['video_id']
            video_title = row['title']

            # Skip verdict is marked as irrelevant or blank
            if 'irrelevant' in row['video_verdict'].lower() or row['video_verdict'] == '':
                print(f"Skip. Not relevant video {index}: {video_title}")
                continue

            # Skip if already analyzed
            if video_id in video_chunks_df['video_id'].values:
                print(f"Skip. Already analyzed video {index}: {video_title}")
                continue

            # Get the transcript
            timestamped_transcript = json.loads(
                ''.join(
                    [row[f'video_transcript_{i}'] for i in range(video_transcript_col_count)]
                )
            )            

            # Submit the task
            future = executor.submit(
                get_yt_chapters_chunks_as_docs,
                video_id,
                video_title,
                timestamped_transcript,
                llm
            )

            # Map the Future to the index
            futures_map[future] = index

        # Collect the results as they complete
        total_tasks = len(futures_map)
        save_interval = 5  # how often to save (in number of completed tasks)
        if total_tasks == 0:
            print('All relevant videos chunked. Skipping this step')
            return

        # Initialize the progress tracker
        progress = SmartProgressBar(total_tasks = total_tasks, description = "Percent complete", save_interval = save_interval)

        # Now, pass only the futures (the keys) to as_completed:
        for future in tqdm(as_completed(futures_map), total=total_tasks):
            index = futures_map[future]  # retrieve the index
            yt_docs = future.result()
            
            if yt_docs:
                # Populate videos research df
                videos_research_df.loc[index, 'chapter_summaries'] = yt_docs[0].metadata.get('chapter_summaries', '')

                # Populate video chunks df
                for doc in yt_docs:
                    # Correct: get the chapter title from the doc metadata
                    chapter_title = doc.metadata.get('chapter_title', '')
                    # Correct: get the channel title from the mapping
                    channel_title = video_id_to_channel_title.get(videos_research_df.loc[index, 'video_id'], row.get('channel_title', 'Unknown'))
                    video_chunks_df = pd.concat([
                        video_chunks_df,
                        pd.DataFrame({
                            'video_id': [videos_research_df.loc[index, 'video_id']],
                            'video_title': [videos_research_df.loc[index, 'title']],
                            'chapter_title': [chapter_title],
                            'metadata': [json.dumps({**doc.metadata, 'channel': channel_title})],
                            'text_0': ['']
                        })
                    ], ignore_index=True)

                    video_chunks_df = create_and_populate_columns(
                        df = video_chunks_df,
                        text = doc.page_content,
                        specific_index = video_chunks_df.shape[0] - 1,
                        col_base_name = "text"
                    )

            # Update progress
            progress.update()

            # Check if we should save
            if progress.should_save():
                print(f'Saving partial progress to sheet after {progress.completed_count} tasks completed.')
                save_to_sheet(worksheet = videos_research_sheet, df = videos_research_df)
                save_to_sheet(worksheet = video_chunks_sheet, df = video_chunks_df)

    # Final save to sheet after all tasks
    print('All rows processed. Saving final DataFrame to sheet.')
    save_to_sheet(worksheet = videos_research_sheet, df = videos_research_df)
    save_to_sheet(worksheet = video_chunks_sheet, df = video_chunks_df)
    
    # Format the newly created sheet
    format_worksheet(video_chunks_sheet)

    return


def delete_video_transcripts(sheet, worksheet_name="Videos Research"):
    """Remove video transcript columns from the Videos Research sheet."""
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    cols = [c for c in df.columns if c.startswith("video_transcript")]
    if cols:
        df = df.drop(columns=cols)
        clear_worksheet(ws)
        save_to_sheet(ws, df)


def delete_video_relevance(sheet, worksheet_name="Videos Research"):
    """Remove analysis and verdict columns from the Videos Research sheet."""
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    cols = [c for c in df.columns if c in ["video_analysis", "video_verdict"]]
    if cols:
        df = df.drop(columns=cols)
        clear_worksheet(ws)
        save_to_sheet(ws, df)


def delete_video_chunks(sheet, videos_research_worksheet="Videos Research", video_chunks_worksheet="Video Chunks"):
    """Remove chapter summaries and delete the Video Chunks sheet."""
    ws, df = get_sheet_data_and_df(sheet, videos_research_worksheet)
    cols = [c for c in df.columns if c.startswith("chapter_summaries")]
    if cols:
        df = df.drop(columns=cols)
        clear_worksheet(ws)
        save_to_sheet(ws, df)
    delete_worksheet(sheet, video_chunks_worksheet)


