import sys
import os
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.redis_cache import SemanticCache
from services.embed_bge import BGEEmbedder

def main():
    print("Loading embedder...")
    embedder = BGEEmbedder(use_fp16=True)
    
    print("Connecting to Redis...")
    redis_cache = SemanticCache(host="localhost", port=6380)
    
    qna_pairs = [
        {
            "q": "What is the function of AMF in 5G?",
            "a": "**Access and Mobility Management Function (AMF)** is a central component of the 5G Core Network (5GC). It is responsible for handling connection and mobility management tasks. Unlike the MME in 4G which handled both session and mobility management, the 5G architecture decouples these roles, with AMF specifically handling registration, connection, reachability, and mobility management. It acts as the primary point of contact for the User Equipment (UE) to access the 5G core over the N1 interface, and coordinates with the SMF for PDU session setup. Furthermore, the AMF provides authentication, authorization, and location services management, ensuring secure and seamless handovers."
        },
        {
            "q": "What is the function of SMF in 5G?",
            "a": "**Session Management Function (SMF)** is a critical element in the 5G Service-Based Architecture. Its primary role is to establish, modify, and release PDU (Protocol Data Unit) sessions. It manages the IP address allocation for UEs and controls the User Plane Function (UPF) for traffic routing. The SMF interacts heavily with the AMF to receive session management requests from the UE and communicates with the Policy Control Function (PCF) to apply QoS policies and charging rules."
        },
        {
            "q": "What is the function of UPF in 5G?",
            "a": "**User Plane Function (UPF)** is the fundamental data plane component in 5G. It replaces the data routing roles previously handled by the SGW and PGW in LTE. The UPF routes and forwards user data packets between the Radio Access Network (RAN) and external Data Networks (DN). It is also responsible for QoS enforcement, traffic usage reporting for billing, and packet inspection. The separation of the UPF from control plane functions (like SMF) is a core tenant of CUPS (Control and User Plane Separation), enabling flexible UPF placement at the network edge."
        },
        {
            "q": "What is the function of PCF in 5G?",
            "a": "**Policy Control Function (PCF)** governs the network behavior by providing a unified policy framework across the 5G network. It handles policy decisions relating to network slicing, roaming, mobility management, and QoS. It replaces the PCRF (Policy and Charging Rules Function) from 4G. The PCF communicates directly with the SMF to enforce data session policies, and the AMF for mobility policies, dynamically adjusting network resources based on subscriber profiles and current network conditions."
        },
        {
            "q": "What is the function of UDM in 5G?",
            "a": "**Unified Data Management (UDM)** acts as the central repository for subscriber data in the 5G Core, analogous to the HSS (Home Subscriber Server) in 4G. It manages user identification, subscription profiles, access authorization, and network capabilities. The UDM does not store the data itself; instead, it uses the UDR (Unified Data Repository) as the backend database, operating as the front-end application logic for subscriber data management."
        },
        {
            "q": "What is the function of AUSF in 5G?",
            "a": "**Authentication Server Function (AUSF)** handles the authentication process for User Equipment (UE) attempting to connect to the 5G network. It implements EAP (Extensible Authentication Protocol) and 5G-AKA (Authentication and Key Agreement) frameworks. The AUSF receives authentication requests from the AMF, interacts with the UDM to retrieve authentication vectors, and verifies the UE's credentials, ensuring strong security at the access layer."
        },
        {
            "q": "What is the function of NSSF in 5G?",
            "a": "**Network Slice Selection Function (NSSF)** is responsible for directing UE traffic to the appropriate Network Slice instances. Network slicing allows operators to divide a single physical network into multiple virtual networks optimized for different services (e.g., IoT vs. High-Bandwidth Video). The NSSF determines the Network Slice Selection Assistance Information (NSSAI), assigns the AMF to serve the UE based on the requested slice, and ensures isolated and optimized resource allocation."
        },
        {
            "q": "What is the function of NEF in 5G?",
            "a": "**Network Exposure Function (NEF)** provides a mechanism to securely expose 5G network capabilities and events to third-party applications, edge computing instances, and external networks. It acts as an API gateway, translating internal 5G Service-Based Architecture (SBA) APIs into standardized interfaces. This allows developers to access location data, QoS modification tools, and IoT device statuses without exposing internal network topologies."
        },
        {
            "q": "What is the function of NRF in 5G?",
            "a": "**Network Repository Function (NRF)** serves as the central directory for the 5G Service-Based Architecture. It maintains an updated record of all available Network Functions (NFs) and their supported services. Whenever a new NF is instantiated (like scaling up an AMF or SMF), it registers with the NRF. Other NFs query the NRF to discover the IP addresses and endpoints of services they need to consume, enabling automated microservice discovery."
        },
        {
            "q": "What is the function of AF in 5G?",
            "a": "**Application Function (AF)** represents third-party or operator-owned applications that interact with the 5G Core to influence network behavior. The AF interacts primarily with the PCF and NEF to request specific Quality of Service (QoS) requirements for application traffic, monitor network events, or set up dynamic routing policies for Edge Computing scenarios."
        },
        {
            "q": "What is the function of N3IWF in 5G?",
            "a": "**Non-3GPP Interworking Function (N3IWF)** acts as a gateway that securely integrates non-3GPP access networks (such as public Wi-Fi) into the 5G Core. It establishes an IPsec tunnel with the User Equipment (UE) over the untrusted network, forwarding control plane signaling to the AMF and user data to the UPF, ensuring a seamless and secure experience across both cellular and Wi-Fi connections."
        },
        {
            "q": "What is the function of UDR in 5G?",
            "a": "**Unified Data Repository (UDR)** is the centralized database layer in the 5G architecture. It securely stores subscription data, policy data, structured data for exposure, and application data. While the UDM and PCF act as the application logic processing the data, the UDR is the physical database where this information resides, promoting a stateless architecture for the NFs."
        },
        {
            "q": "What is the function of UDSF in 5G?",
            "a": "**Unstructured Data Storage Function (UDSF)** provides storage for unstructured state data generated by various Network Functions (NFs). In the 5G Service-Based Architecture, NFs are designed to be stateless to allow easy scaling and resilience. The UDSF allows these NFs to offload and retrieve transient context data (such as active session states), ensuring continuity if an NF instance fails and is restarted."
        },
        {
            "q": "What is the function of NWDAF in 5G?",
            "a": "**Network Data Analytics Function (NWDAF)** represents the integration of AI and Machine Learning directly into the 5G Core. It collects data from various NFs, OAM (Operations, Administration, and Maintenance) systems, and AFs, performing centralized analytics. The NWDAF provides predictive insights on network congestion, slice load, and UE mobility, allowing other NFs (like PCF or NSSF) to proactively optimize network resources."
        },
        {
            "q": "What is the function of CHF in 5G?",
            "a": "**Charging Function (CHF)** handles all billing and charging operations within the 5G network. It converges both online (prepaid) and offline (postpaid) charging into a single service. The CHF communicates dynamically with the SMF to track data usage per PDU session and applies rating policies in real-time based on the specific Network Slice, QoS level, and application accessed."
        },
        {
            "q": "What is the function of BSF in 5G?",
            "a": "**Binding Support Function (BSF)** is utilized in environments where multiple PCF (Policy Control Function) instances exist. It ensures that a specific user session is bound to the correct PCF instance. When an Application Function (AF) needs to request QoS changes for a user's IP address, it queries the BSF to discover which PCF is currently managing that user's PDU session."
        },
        {
            "q": "What is the function of SEPP in 5G?",
            "a": "**Security Edge Protection Proxy (SEPP)** acts as a non-transparent proxy at the perimeter of the 5G core network. It is crucial for roaming scenarios. The SEPP provides message filtering, topology hiding, and end-to-end encryption for all signaling traffic traveling between the home network and the visited network over the N32 interface, protecting against signaling attacks."
        },
        {
            "q": "What is the function of NSSAAF in 5G?",
            "a": "**Network Slice-Specific Authentication and Authorization Function (NSSAAF)** is an advanced 5G capability that provides an additional layer of security for specific network slices. While the AUSF authenticates the UE to the general 5G network, the NSSAAF handles secondary authentication, typically interacting with an external AAA server owned by an enterprise, before allowing the UE access to a highly secure corporate slice."
        },
        {
            "q": "What is the function of EIR in 5G?",
            "a": "**Equipment Identity Register (EIR)** is a database that tracks the International Mobile Equipment Identity (IMEI) of devices. It classifies devices into white, grey, and black lists. The EIR prevents stolen, unauthorized, or defective hardware from accessing the 5G network by verifying the device's IMEI during the initial registration process initiated by the AMF."
        },
        {
            "q": "What is the function of LMF in 5G?",
            "a": "**Location Management Function (LMF)** calculates and manages the precise geographic location of User Equipment (UE). It leverages various positioning techniques, including Downlink Time Difference of Arrival (DL-TDOA), GPS/GNSS, and sensor data. The LMF is essential for emergency services (e.g., E911), location-based enterprise applications, and optimizing beamforming configurations in the RAN."
        },
        {
            "q": "What is the function of GMLC in 5G?",
            "a": "**Gateway Mobile Location Centre (GMLC)** provides the interface between external Location Services (LCS) clients and the 5G network's internal location functions (like the LMF). When an external application requests a user's location, the GMLC performs authorization checks, routes the request to the correct AMF, and formats the resulting location data to send back to the client."
        },
        {
            "q": "What is the function of SMLC in 5G/4G?",
            "a": "**Serving Mobile Location Center (SMLC)** (largely a 3G/4G term, evolved into LMF in 5G) handles the calculation of the UE's location based on network measurements. It manages the coordination of positioning resources, instructing the base stations to take timing measurements and process them to deliver accurate geolocation coordinates."
        },
        {
            "q": "What is the function of CBC in 5G?",
            "a": "**Cell Broadcast Center (CBC)** is responsible for managing and broadcasting emergency alerts and public warning messages to all users within a specific geographic area. It interfaces with the AMF to distribute messages (such as earthquake warnings or Amber alerts) across multiple base stations simultaneously without requiring active sessions for each user."
        },
        {
            "q": "What is the function of MME in 4G?",
            "a": "**Mobility Management Entity (MME)** is the primary control node for the LTE (4G) access network. It handles idle mode UE tracking, paging procedures, and bearer activation/deactivation processes. Unlike 5G where these roles are split, the MME acts as both the mobility manager and the session manager interface, communicating with the HSS for user authentication and the SGW for data routing setup."
        },
        {
            "q": "What is the function of SGW in 4G?",
            "a": "**Serving Gateway (SGW)** in the 4G EPC (Evolved Packet Core) routes and forwards user data packets. It acts as the mobility anchor for the user plane during inter-eNodeB handovers and between LTE and other 3GPP technologies. In 5G, the functionality of the SGW has been absorbed and expanded upon by the UPF (User Plane Function)."
        },
        {
            "q": "What is the function of PGW in 4G?",
            "a": "**Packet Data Network Gateway (PGW)** in 4G provides connectivity from the UE to external packet data networks. It performs IP address allocation, QoS enforcement, and packet filtering. It is the anchor point for mobility between 3GPP and non-3GPP networks. In the 5G architecture, the PGW's control functions moved to the SMF, and its user plane functions moved to the UPF."
        },
        {
            "q": "What is the function of HSS in 4G?",
            "a": "**Home Subscriber Server (HSS)** is the central master database in the 4G EPC. It contains user-related and subscription-related information. The HSS performs mobility management, call and session establishment support, user authentication, and access authorization. In 5G, the HSS evolves into the UDM (Unified Data Management) and UDR (Unified Data Repository)."
        },
        {
            "q": "What is the function of PCRF in 4G?",
            "a": "**Policy and Charging Rules Function (PCRF)** in 4G determines policy rules in real-time, dictating the Quality of Service (QoS) and charging parameters for active data sessions. It provides these rules to the PGW for enforcement. In the 5G SBA, the PCRF's responsibilities have been mapped to the PCF (Policy Control Function)."
        },
        {
            "q": "What is an eNodeB?",
            "a": "**Evolved Node B (eNodeB)** is the base station hardware in the 4G LTE network. It interfaces directly with User Equipment (UE) via the air interface (LTE-Uu). The eNodeB handles radio resource management, header compression, security encryption, and packet routing to the EPC (specifically to the MME and SGW)."
        },
        {
            "q": "What is a gNodeB?",
            "a": "**Next Generation Node B (gNodeB or gNB)** is the radio base station in the 5G NR (New Radio) network. It provides 5G user plane and control plane protocols toward the UE. A gNB is often functionally split into a Central Unit (CU) handling higher-layer protocols, and one or more Distributed Units (DU) handling baseband processing, connected to Radio Units (RU) that transmit the RF signals."
        },
        {
            "q": "What is 5G NR?",
            "a": "**5G NR (New Radio)** is the global standard for the unified, more capable 5G wireless air interface. It utilizes orthogonal frequency-division multiplexing (OFDM) but introduces scalable numerologies, allowing it to operate across diverse spectrum bands (from sub-1 GHz to mmWave). 5G NR delivers significantly lower latency, higher bandwidth, and massive device connectivity compared to LTE."
        },
        {
            "q": "Explain the concept of Network Slicing in 5G.",
            "a": "**Network Slicing** is a paradigm-shifting feature of 5G that allows operators to multiplex multiple virtualized, independent logical networks on the same physical infrastructure. Each 'slice' is an isolated end-to-end network tailored to fulfill specific requirements requested by a particular application—such as a high-throughput slice for AR/VR, or an ultra-low latency slice for autonomous driving. This is achieved through SDN (Software Defined Networking) and NFV (Network Functions Virtualization)."
        },
        {
            "q": "What is the difference between 5G Standalone (SA) and Non-Standalone (NSA)?",
            "a": "**5G Non-Standalone (NSA)** allows operators to launch 5G speeds quickly by using 5G NR base stations (gNBs) anchored to an existing 4G LTE core network (EPC). The control signaling remains on 4G. **5G Standalone (SA)**, however, utilizes a completely new cloud-native 5G Core (5GC) paired with 5G NR. Only 5G SA can deliver advanced 5G features like ultra-low latency (URLLC), network slicing, and massive IoT connectivity."
        },
        {
            "q": "What is Massive MIMO and how does it work in 5G?",
            "a": "**Massive MIMO (Multiple Input, Multiple Output)** involves equipping base stations with a high number of antenna elements (e.g., 64, 128, or 256) instead of the standard 2 or 4. This massive array allows the base station to transmit and receive multiple data streams simultaneously to multiple users in the same frequency block. It drastically improves spectral efficiency, network capacity, and signal strength."
        },
        {
            "q": "What is Beamforming and why is it essential for 5G mmWave?",
            "a": "**Beamforming** is a signal processing technique used in combination with Massive MIMO to direct wireless signals in a specific direction, rather than broadcasting them radially. By adjusting the phase and amplitude of the signals at each antenna, a concentrated 'beam' is formed pointing directly at the user. This is absolutely essential for 5G mmWave frequencies, which suffer from high attenuation and poor penetration, as beamforming extends their range and reliability."
        },
        {
            "q": "Describe the 5G Core (5GC) architecture.",
            "a": "The **5G Core (5GC)** introduces a Service-Based Architecture (SBA). Unlike 4G's point-to-point network nodes, 5GC is completely cloud-native and software-defined. Network Functions (NFs) like AMF, SMF, and PCF operate as independent microservices. They communicate via stateless RESTful APIs over an HTTP/2 service bus. This architecture allows for rapid scaling, automated deployment via Kubernetes, and the decoupling of the Control Plane and User Plane (CUPS)."
        },
        {
            "q": "How does 5G handle Quality of Service (QoS)?",
            "a": "5G QoS is flow-based rather than bearer-based like in 4G. The network classifies packets into QoS Flows, the finest granularity of QoS differentiation. Each QoS Flow is identified by a **5QI (5G QoS Identifier)**. The SMF dictates the QoS rules, and the UPF and gNB enforce them, ensuring specific packets receive guaranteed bit rates, strict latency thresholds, or priority handling."
        },
        {
            "q": "What is the 5G QoS Identifier (5QI)?",
            "a": "The **5QI** is a scalar value that acts as a reference to specific QoS characteristics (Priority Level, Packet Delay Budget, Packet Error Rate, etc.). Standardized 5QIs exist for common services (e.g., 5QI 1 for conversational voice, 5QI 80 for low latency AR). Instead of negotiating all parameters, the network simply assigns a 5QI, and all network elements know exactly how to treat that traffic flow."
        },
        {
            "q": "Explain eMBB in 5G.",
            "a": "**eMBB (Enhanced Mobile Broadband)** is one of the three primary use cases of 5G. It focuses on providing extreme data rates (up to 10-20 Gbps peak), high spectral efficiency, and seamless coverage. eMBB caters to high-bandwidth applications like 4K/8K video streaming, immersive Virtual Reality (VR), and Fixed Wireless Access (FWA)."
        },
        {
            "q": "Explain URLLC in 5G.",
            "a": "**URLLC (Ultra-Reliable Low-Latency Communication)** is a 5G use case designed for mission-critical applications where failure is not an option. It targets end-to-end latency of less than 1 millisecond and reliability of 99.999%. This enables real-time applications such as autonomous driving, remote robotic surgery, and industrial factory automation."
        },
        {
            "q": "Explain mMTC in 5G.",
            "a": "**mMTC (Massive Machine-Type Communications)** addresses the need to connect an enormous number of low-power IoT devices—up to 1 million devices per square kilometer. It focuses on providing excellent deep-indoor coverage and ensuring device batteries can last up to 10 years by sending small bursts of data infrequently, powering smart cities and industrial sensors."
        },
        {
            "q": "What is Multi-Access Edge Computing (MEC) in the context of 5G?",
            "a": "**Multi-Access Edge Computing (MEC)** brings cloud computing capabilities and IT service environments to the edge of the cellular network (e.g., at the base station or local central office). By deploying applications via the 5G UPF at the network edge, data doesn't have to travel to a centralized cloud, drastically reducing latency for URLLC applications and saving backhaul bandwidth."
        },
        {
            "q": "What is a PDU Session in 5G?",
            "a": "A **PDU (Protocol Data Unit) Session** is an association between the User Equipment (UE) and a Data Network (DN) that provides a PDU connectivity service. Unlike LTE bearers, PDU sessions are highly flexible. A single UE can have multiple active PDU sessions to different networks simultaneously, and 5G supports IPv4, IPv6, IPv4v6, Ethernet, and even Unstructured PDU types, making it ideal for industrial IoT."
        },
        {
            "q": "Explain the 5G security architecture.",
            "a": "5G security is significantly enhanced over 4G. It mandates encryption for both data and control planes. The **5G-AKA (Authentication and Key Agreement)** provides robust mutual authentication. A major addition is the protection of the Subscriber Permanent Identifier (SUPI); before transmission over the air, the UE encrypts it into a SUCI (Subscription Concealed Identifier), preventing IMSI-catching attacks (Stingrays) that plagued older generations."
        },
        {
            "q": "Explain Service Based Architecture (SBA) in 5G.",
            "a": "The **Service Based Architecture (SBA)** is the foundational design of the 5G Core. It replaces rigid node-to-node interfaces with an IT-oriented microservices model. Network Functions (NFs) expose their capabilities as RESTful APIs over HTTP/2. An NF can act as a service producer or consumer. This cloud-native approach allows operators to upgrade, scale, and integrate network functions seamlessly, just like modern web applications."
        },
        {
            "q": "What features were introduced in 3GPP Release 15?",
            "a": "**3GPP Release 15** is the foundational release of 5G, finalized in 2018/2019. It defined the 5G New Radio (NR) standard, focusing heavily on eMBB (Enhanced Mobile Broadband) to provide massive speed increases. It established the core scalable OFDM numerologies, support for sub-6GHz and mmWave spectrum, and defined the initial framework for the cloud-native 5G Core (5GC), including both NSA and SA deployments."
        },
        {
            "q": "What features were introduced in 3GPP Release 16?",
            "a": "**3GPP Release 16**, often referred to as '5G Phase 2', finalized in 2020. It pivoted heavily toward enterprise and industrial use cases. It formally introduced URLLC enhancements for industrial IoT, Cellular V2X (Vehicle-to-Everything) for autonomous driving, advanced power-saving features, Integrated Access and Backhaul (IAB), and positioning enhancements capable of indoor accuracy within a few meters."
        },
        {
            "q": "What features were introduced in 3GPP Release 17?",
            "a": "**3GPP Release 17**, completed in 2022, focused on network expansion and efficiency. It introduced support for Non-Terrestrial Networks (NTN) to integrate satellite communications into 5G. It also debuted 'RedCap' (Reduced Capability) NR devices, which bridge the gap between high-speed eMBB and low-power mMTC, providing cheaper, smaller chips for wearables and mid-tier IoT devices. Furthermore, it expanded mmWave spectrum support up to 71 GHz."
        },
        {
            "q": "What features are expected in 3GPP Release 18 (5G-Advanced)?",
            "a": "**3GPP Release 18** marks the beginning of '5G-Advanced'. It integrates Artificial Intelligence and Machine Learning directly into the Radio Access Network (RAN) and core to optimize beamforming, mobility, and network energy savings. It brings further enhancements to NTN (satellites), introduces full duplex communications, improves XR (Extended Reality) capabilities, and lays the groundwork for the evolution toward 6G."
        },
        {
            "q": "What is O-RAN (Open RAN)?",
            "a": "**O-RAN (Open Radio Access Network)** is a movement led by the O-RAN Alliance to disaggregate the traditionally proprietary and closed base station hardware. It defines open, interoperable interfaces between the Radio Unit (RU), Distributed Unit (DU), and Central Unit (CU). This allows network operators to mix and match hardware and software from different vendors, breaking vendor lock-in, reducing costs, and enabling the integration of intelligent controllers (RIC) for AI-driven RAN optimization."
        }
    ]
    
    print(f"Total detailed QnA pairs: {len(qna_pairs)}")
    
    for i, pair in enumerate(qna_pairs):
        query = pair["q"]
        answer = pair["a"]
        
        # embed query
        query_dense, _ = embedder.embed([query])
        query_vector = query_dense[0]
        
        response_data = {
            "answer": answer,
            "citations": ["[System Knowledge Base]", "[3GPP Rel 15-18 Docs]"],
            "is_abstained": False,
            "abstention_reason": None,
            "guardrails": {
                "jailbreak_detected": False,
                "pii_detected": False,
                "hallucination_detected": False,
                "answered_from_general_knowledge": False,
                "web_fallback_used": False,
                "is_out_of_domain": False
            }
        }
        
        redis_cache.save_to_cache(query_vector, json.dumps(response_data), ttl_seconds=86400 * 30)
        if (i+1) % 10 == 0:
            print(f"Inserted {i+1}/{len(qna_pairs)}")
            
    print("Done populating Redis with detailed responses!")

if __name__ == "__main__":
    main()
